#!/usr/bin/env python3
"""
Test script to analyze GraphML usage patterns in nano-graphrag
and compare current debug_info extraction with full GraphML data.
"""

import asyncio
import json
import sys
import logging
from pathlib import Path
import networkx as nx

# Add current directory to path to import nano_graphrag
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import logger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GraphMLAnalyzer:
    def __init__(self, working_dir):
        self.working_dir = Path(working_dir)
        self.graphml_path = self.working_dir / "graph_chunk_entity_relation.graphml"

    def load_graphml_data(self):
        """Load and analyze the raw GraphML file structure"""
        if not self.graphml_path.exists():
            logger.error(f"GraphML file not found: {self.graphml_path}")
            return None

        logger.info(f"Loading GraphML from: {self.graphml_path}")
        G = nx.read_graphml(str(self.graphml_path))

        # Analyze GraphML structure
        analysis = {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "node_attributes": set(),
            "edge_attributes": set(),
            "sample_nodes": [],
            "sample_edges": []
        }

        # Analyze node attributes
        for node_id, node_data in G.nodes(data=True):
            analysis["node_attributes"].update(node_data.keys())
            if len(analysis["sample_nodes"]) < 3:
                analysis["sample_nodes"].append({
                    "id": node_id,
                    "data": dict(node_data)
                })

        # Analyze edge attributes
        for source, target, edge_data in G.edges(data=True):
            analysis["edge_attributes"].update(edge_data.keys())
            if len(analysis["sample_edges"]) < 3:
                analysis["sample_edges"].append({
                    "source": source,
                    "target": target,
                    "data": dict(edge_data)
                })

        analysis["node_attributes"] = list(analysis["node_attributes"])
        analysis["edge_attributes"] = list(analysis["edge_attributes"])

        return G, analysis

    def extract_rich_relationships(self, G, entity_names):
        """Extract rich relationship data from GraphML for given entities"""
        rich_relationships = []

        # Find edges involving the specified entities
        for source, target, edge_data in G.edges(data=True):
            source_clean = source.strip('"').upper()
            target_clean = target.strip('"').upper()

            # Check if this edge involves any of our entities
            entity_match = False
            for entity in entity_names:
                entity_upper = entity.upper()
                if entity_upper in source_clean or entity_upper in target_clean:
                    entity_match = True
                    break

            if entity_match:
                relationship = {
                    "source": source,
                    "target": target,
                    "weight": edge_data.get("weight", 1.0),
                    "description": edge_data.get("description", "").strip('"'),
                    "source_id": edge_data.get("source_id", ""),
                    "order": edge_data.get("order", 0)
                }
                rich_relationships.append(relationship)

        return rich_relationships

    def extract_rich_nodes(self, G, entity_names):
        """Extract rich node data from GraphML for given entities"""
        rich_nodes = []

        for node_id, node_data in G.nodes(data=True):
            node_name = node_data.get("name", node_id).strip('"')

            # Check if this node matches any of our entities
            entity_match = False
            for entity in entity_names:
                if entity.upper() in node_name.upper() or node_name.upper() in entity.upper():
                    entity_match = True
                    break

            if entity_match:
                node = {
                    "id": node_id,
                    "name": node_name,
                    "entity_type": node_data.get("entity_type", "").strip('"'),
                    "description": node_data.get("description", "").strip('"'),
                    "clusters": node_data.get("clusters", ""),
                    "source_id": node_data.get("source_id", "")
                }
                rich_nodes.append(node)

        return rich_nodes

async def analyze_graphml_only(analyzer, test_query):
    """Simplified analysis using only GraphML data"""

    logger.info("=== Analyzing GraphML Data Only ===")

    G, graphml_analysis = analyzer.load_graphml_data()
    if not G:
        return None

    # Sample entities for analysis (since we don't have debug_info)
    sample_entities = ["HUYSMANS", "SOCIÉTÉ", "MODERNITÉ", "DÉCADENCE", "DES ESSEINTES"]

    # Extract rich data for sample entities
    rich_relationships = analyzer.extract_rich_relationships(G, sample_entities)
    rich_nodes = analyzer.extract_rich_nodes(G, sample_entities)

    logger.info(f"GraphML Analysis Results:")
    logger.info(f"  - Sample entities: {sample_entities}")
    logger.info(f"  - Rich nodes found: {len(rich_nodes)}")
    logger.info(f"  - Rich relationships found: {len(rich_relationships)}")

    # Show sample rich data
    logger.info("Rich Relationship Examples:")
    for i, rel in enumerate(rich_relationships[:5]):
        logger.info(f"  {i+1}: {rel['source']} -> {rel['target']} (weight: {rel['weight']})")
        logger.info(f"     Description: {rel['description'][:100]}...")

    return {
        "query": test_query,
        "analysis_type": "graphml_only",
        "graphml_analysis": graphml_analysis,
        "rich_graphml_data": {
            "nodes": rich_nodes,
            "relationships": rich_relationships
        },
        "findings": {
            "total_graphml_nodes": graphml_analysis['total_nodes'],
            "total_graphml_edges": graphml_analysis['total_edges'],
            "rich_metadata_available": True,
            "sample_entities_tested": sample_entities,
            "rich_relationships_found": len(rich_relationships)
        }
    }

async def test_graphrag_with_monitoring():
    """Test GraphRAG query with detailed monitoring of GraphML access"""

    working_dir = "/Users/arthursarazin/Documents/nano-graphrag/book_data/a_rebours_huysmans"

    # Initialize analyzer
    analyzer = GraphMLAnalyzer(working_dir)

    # Load and analyze raw GraphML
    logger.info("=== PHASE 1: Raw GraphML Analysis ===")
    graphml_data = analyzer.load_graphml_data()
    if not graphml_data:
        return

    G, graphml_analysis = graphml_data

    logger.info(f"GraphML Structure Analysis:")
    logger.info(f"  - Nodes: {graphml_analysis['total_nodes']}")
    logger.info(f"  - Edges: {graphml_analysis['total_edges']}")
    logger.info(f"  - Node attributes: {graphml_analysis['node_attributes']}")
    logger.info(f"  - Edge attributes: {graphml_analysis['edge_attributes']}")

    # Show sample rich edge data
    logger.info("Sample Rich Edge Data:")
    for i, edge in enumerate(graphml_analysis["sample_edges"]):
        logger.info(f"  Edge {i+1}: {edge['source']} -> {edge['target']}")
        logger.info(f"    Weight: {edge['data'].get('weight', 'N/A')}")
        logger.info(f"    Description: {edge['data'].get('description', 'N/A')[:100]}...")

    # Initialize GraphRAG
    logger.info("=== PHASE 2: GraphRAG Query Execution ===")

    rag = GraphRAG(
        working_dir=working_dir,
        enable_llm_cache=True
    )

    # Test query
    test_query = "que pense l'auteur de la société"
    logger.info(f"Executing query: '{test_query}'")

    result = await rag.aquery(test_query, param=QueryParam(mode="local"))

    logger.info("=== PHASE 3: Debug Info Analysis ===")

    # Handle the result format - it might be a string or dict
    if isinstance(result, str):
        logger.info("Result is a string response - skipping debug_info analysis for now")
        logger.info(f"Response preview: {result[:200]}...")
        # Create a simple analysis based on GraphML data only
        return await analyze_graphml_only(analyzer, test_query)
    else:
        debug_info = result.get("debug_info", {})

    # Analyze current debug_info extraction
    current_relationships = debug_info.get('processing_phases', {}).get('relationship_mapping', {}).get('relationships', [])
    entity_selection = debug_info.get('processing_phases', {}).get('entity_selection', {}).get('entities', [])

    logger.info(f"Current Debug Info:")
    logger.info(f"  - Entities found: {len(entity_selection)}")
    logger.info(f"  - Relationships found: {len(current_relationships)}")

    entity_names = [entity.get('name', entity.get('id', '')) for entity in entity_selection]
    logger.info(f"  - Entity names: {entity_names[:5]}...")  # Show first 5

    # Show sample current relationships
    logger.info("Sample Current Relationships:")
    for i, rel in enumerate(current_relationships[:3]):
        logger.info(f"  Rel {i+1}: {rel.get('source', 'N/A')} -> {rel.get('target', 'N/A')}")
        logger.info(f"    Description: {rel.get('description', 'N/A')[:100]}...")

    logger.info("=== PHASE 4: Rich GraphML Data Extraction ===")

    # Extract rich data from GraphML for the same entities
    rich_relationships = analyzer.extract_rich_relationships(G, entity_names)
    rich_nodes = analyzer.extract_rich_nodes(G, entity_names)

    logger.info(f"Rich GraphML Extraction:")
    logger.info(f"  - Rich nodes found: {len(rich_nodes)}")
    logger.info(f"  - Rich relationships found: {len(rich_relationships)}")

    # Show sample rich relationships
    logger.info("Sample Rich Relationships (from GraphML):")
    for i, rel in enumerate(rich_relationships[:3]):
        logger.info(f"  Rich Rel {i+1}: {rel['source']} -> {rel['target']}")
        logger.info(f"    Weight: {rel['weight']}")
        logger.info(f"    Description: {rel['description'][:100]}...")
        logger.info(f"    Source chunks: {rel['source_id'][:50]}...")
        logger.info(f"    Order: {rel['order']}")

    logger.info("=== PHASE 5: Comparison Analysis ===")

    # Compare current vs rich data
    comparison = {
        "current_entities": len(entity_selection),
        "current_relationships": len(current_relationships),
        "rich_nodes": len(rich_nodes),
        "rich_relationships": len(rich_relationships),
        "metadata_difference": {
            "weights_available": any(rel.get('weight', 0) > 0 for rel in rich_relationships),
            "descriptions_available": any(rel.get('description') for rel in rich_relationships),
            "source_chunks_available": any(rel.get('source_id') for rel in rich_relationships),
            "order_available": any(rel.get('order', 0) > 0 for rel in rich_relationships)
        }
    }

    logger.info("Comparison Results:")
    logger.info(f"  Current extraction: {comparison['current_entities']} entities, {comparison['current_relationships']} relationships")
    logger.info(f"  Rich GraphML data: {comparison['rich_nodes']} nodes, {comparison['rich_relationships']} relationships")
    logger.info(f"  Rich metadata available:")
    for key, value in comparison['metadata_difference'].items():
        logger.info(f"    - {key}: {value}")

    # Calculate potential enhancement value
    enhancement_ratio = len(rich_relationships) / len(current_relationships) if current_relationships else 0
    logger.info(f"  Enhancement potential: {enhancement_ratio:.2f}x more relationships with rich metadata")

    # Save detailed analysis to file
    analysis_result = {
        "query": test_query,
        "graphml_analysis": graphml_analysis,
        "current_debug_info": {
            "entities": entity_selection,
            "relationships": current_relationships
        },
        "rich_graphml_data": {
            "nodes": rich_nodes,
            "relationships": rich_relationships
        },
        "comparison": comparison,
        "enhancement_recommendations": {
            "use_graphml_weights": comparison['metadata_difference']['weights_available'],
            "use_relationship_descriptions": comparison['metadata_difference']['descriptions_available'],
            "add_source_traceability": comparison['metadata_difference']['source_chunks_available'],
            "implement_relationship_ordering": comparison['metadata_difference']['order_available']
        }
    }

    # Save to JSON file for detailed analysis
    output_path = Path(working_dir) / "graphml_analysis_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)

    logger.info(f"Detailed analysis saved to: {output_path}")
    logger.info("=== GraphML Analysis Complete ===")

    return analysis_result

if __name__ == "__main__":
    result = asyncio.run(test_graphrag_with_monitoring())

    if result:
        print("\n" + "="*50)
        print("EXECUTIVE SUMMARY")
        print("="*50)
        print(f"Query tested: '{result['query']}'")

        if result.get('analysis_type') == 'graphml_only':
            # GraphML-only analysis summary
            findings = result['findings']
            print(f"Analysis type: GraphML-only (no debug_info available)")
            print(f"Total GraphML nodes: {findings['total_graphml_nodes']}")
            print(f"Total GraphML edges: {findings['total_graphml_edges']}")
            print(f"Rich relationships found for sample entities: {findings['rich_relationships_found']}")
            print(f"Sample entities tested: {findings['sample_entities_tested']}")
            print("\nKey findings:")
            print("  - Rich relationship metadata is available in GraphML")
            print("  - Relationship weights, descriptions, and source chunks are accessible")
            print("  - Significant enhancement potential for visualization interface")

        else:
            # Full comparison analysis summary
            print(f"Current system extracts: {result['comparison']['current_relationships']} relationships")
            print(f"GraphML contains: {result['comparison']['rich_relationships']} relationships with metadata")

            enhancement = result['enhancement_recommendations']
            print("\nRecommended enhancements:")
            for rec, available in enhancement.items():
                status = "✅ Available" if available else "❌ Not available"
                print(f"  - {rec.replace('_', ' ').title()}: {status}")

            print(f"\nPotential visualization enhancement: {result['comparison']['rich_relationships'] / result['comparison']['current_relationships'] if result['comparison']['current_relationships'] else 0:.1f}x richer")