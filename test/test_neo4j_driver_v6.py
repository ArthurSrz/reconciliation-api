#!/usr/bin/env python3
"""
Test Neo4j Driver v6.0.2 (installed with MCP package)
This version might have better connection handling
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load credentials
load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI', 'neo4j+s://f768707e.databases.neo4j.io')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')

print("=" * 70)
print("Testing Neo4j Driver v6.0.2")
print("=" * 70)
print(f"URI: {NEO4J_URI}")
print(f"User: {NEO4J_USER}")
print(f"Database: {NEO4J_DATABASE}\n")

# Try different configuration options
configs = [
    {
        "name": "Standard config (default)",
        "options": {}
    },
    {
        "name": "With explicit trusted certs",
        "options": {"trusted_certificates": None}
    },
    {
        "name": "With SSL disabled (last resort)",
        "options": {"encrypted": False}
    }
]

for config in configs:
    print(f"\n{'=' * 70}")
    print(f"Testing: {config['name']}")
    print("-" * 70)

    try:
        # Create driver with specific config
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            **config['options']
        )

        print("✅ Driver created")

        # Verify connectivity
        print("Attempting to verify connectivity...")
        driver.verify_connectivity()

        print("✅✅ CONNECTIVITY VERIFIED!")
        print("\n🎉 SUCCESS! Connection to Neo4j Aura works!")

        # Try a simple query
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("RETURN 1 as test, 'Connection successful!' as message")
            record = result.single()
            print(f"\n📊 Test query result:")
            print(f"   test = {record['test']}")
            print(f"   message = {record['message']}")

            # Get database info
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            for record in result:
                print(f"\n📦 Database info:")
                print(f"   Name: {record['name']}")
                print(f"   Edition: {record['edition']}")
                print(f"   Versions: {record['versions']}")

            # Try to search for data
            print(f"\n🔍 Searching for nodes in database...")
            result = session.run("MATCH (n) RETURN count(n) as total_nodes LIMIT 1")
            record = result.single()
            total = record['total_nodes']
            print(f"   Total nodes in database: {total}")

            if total > 0:
                # Get some sample nodes
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n) as labels, properties(n) as props
                    LIMIT 5
                """)
                print(f"\n   Sample nodes:")
                for i, record in enumerate(result, 1):
                    print(f"   {i}. Labels: {record['labels']}")
                    print(f"      Properties: {list(record['props'].keys())[:3]}...")

                # Try to search for "La Promesse de l'aube"
                print(f"\n🎯 Searching for 'promesse' or 'aube'...")
                result = session.run("""
                    MATCH (n)
                    WHERE toLower(toString(n.name)) CONTAINS 'promesse'
                       OR toLower(toString(n.name)) CONTAINS 'aube'
                       OR toLower(toString(n.title)) CONTAINS 'promesse'
                       OR toLower(toString(n.title)) CONTAINS 'aube'
                    RETURN n
                    LIMIT 10
                """)

                found = list(result)
                if found:
                    print(f"   ✅ Found {len(found)} nodes related to 'La Promesse de l'aube'!")
                    for i, record in enumerate(found[:3], 1):
                        node = record['n']
                        print(f"\n   Node {i}:")
                        print(f"      Labels: {list(node.labels)}")
                        print(f"      Properties: {dict(node)}")
                else:
                    print(f"   ℹ️  No nodes found matching 'promesse' or 'aube'")
                    print(f"   (The database might not have data for this book yet)")

        driver.close()
        print("\n" + "=" * 70)
        print("✅✅✅ CONNECTION TEST: COMPLETE SUCCESS!")
        print("=" * 70)
        print("\nThis configuration works! Neo4j Aura is accessible!")
        break  # Stop on first success

    except Exception as e:
        print(f"❌ Failed: {str(e)[:200]}")
        print(f"   Error type: {type(e).__name__}")
        continue

print("\n" + "=" * 70)
print("Test completed.")
print("=" * 70)
