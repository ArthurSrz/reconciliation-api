#!/usr/bin/env python3
"""Test different Neo4j connection methods"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')
AURA_INSTANCE = "f768707e.databases.neo4j.io"

# Different connection URIs to test
connection_methods = [
    ("neo4j+s://", f"neo4j+s://{AURA_INSTANCE}"),
    ("neo4j+ssc://", f"neo4j+ssc://{AURA_INSTANCE}"),
    ("bolt+s://", f"bolt+s://{AURA_INSTANCE}"),
    ("bolt+ssc://", f"bolt+ssc://{AURA_INSTANCE}"),
    ("neo4j://", f"neo4j://{AURA_INSTANCE}"),
    ("bolt://", f"bolt://{AURA_INSTANCE}"),
]

print("Testing different Neo4j connection methods...\n")
print(f"Username: {NEO4J_USER}")
print(f"Instance: {AURA_INSTANCE}\n")
print("=" * 70)

for method_name, uri in connection_methods:
    print(f"\nTesting: {method_name}")
    print(f"URI: {uri}")
    print("-" * 70)

    try:
        # Create driver
        driver = GraphDatabase.driver(
            uri,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        # Try to verify connectivity
        driver.verify_connectivity()

        print(f"✅ SUCCESS! Connection established with {method_name}")

        # Try a simple query
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            print(f"   Test query result: {record['test']}")

            # Get some database info
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            for record in result:
                print(f"   Neo4j {record['edition']}: {record['name']} - {record['versions']}")

        driver.close()
        print(f"\n🎉 {method_name} WORKS! Use this URI format.\n")
        break  # Stop at first successful connection

    except Exception as e:
        print(f"❌ FAILED: {str(e)[:100]}")
        continue

print("\n" + "=" * 70)
print("Connection test completed.")
