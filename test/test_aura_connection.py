#!/usr/bin/env python3
"""
Test Neo4j Aura connection with the correct credentials
Instance: f768707e.databases.neo4j.io
"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Neo4j Aura credentials from Railway environment
NEO4J_URI = "neo4j+s://f768707e.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "HdHTvHXykt-ueOuz186XtkWNHsQ4kXvHFZocXGvolng"
NEO4J_DATABASE = "neo4j"

print("🔧 Testing Neo4j Aura Connection")
print("=" * 50)
print(f"URI: {NEO4J_URI}")
print(f"User: {NEO4J_USER}")
print(f"Database: {NEO4J_DATABASE}")
print("=" * 50)

try:
    # Create driver
    print("\n1️⃣ Creating Neo4j driver...")
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    # Verify connectivity
    print("2️⃣ Verifying connectivity...")
    driver.verify_connectivity()
    print("✅ Connection verified!")

    # Test basic query
    print("\n3️⃣ Testing basic query...")
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run("RETURN 1 as test, 'Hello Neo4j!' as message")
        record = result.single()
        print(f"✅ Test query result: {record['test']} - {record['message']}")

        # Get database info
        print("\n4️⃣ Getting database information...")
        result = session.run("CALL dbms.components() YIELD name, versions, edition")
        for record in result:
            print(f"✅ Neo4j {record['edition']}: {record['name']} - {record['versions']}")

        # Check if there's any data
        print("\n5️⃣ Checking for existing data...")
        result = session.run("MATCH (n) RETURN count(n) as node_count LIMIT 1")
        count_record = result.single()
        node_count = count_record['node_count']
        print(f"✅ Total nodes in database: {node_count}")

        if node_count > 0:
            # Get a sample of node types
            print("\n6️⃣ Getting sample node types...")
            result = session.run("""
                MATCH (n)
                WITH DISTINCT labels(n) as node_labels, count(n) as count
                RETURN node_labels, count
                ORDER BY count DESC
                LIMIT 5
            """)
            for record in result:
                labels = record['node_labels']
                count = record['count']
                print(f"   - {labels}: {count} nodes")

            # Get a sample node
            print("\n7️⃣ Getting sample node...")
            result = session.run("MATCH (n) RETURN n LIMIT 1")
            sample_record = result.single()
            if sample_record:
                sample_node = sample_record['n']
                print(f"   - Sample node labels: {list(sample_node.labels)}")
                print(f"   - Sample node properties: {dict(sample_node)}")

    driver.close()
    print("\n🎉 SUCCESS! Neo4j Aura connection is working perfectly!")
    print("\n✅ Ready to use in Reconciliation API")

except Exception as e:
    print(f"\n❌ CONNECTION FAILED: {str(e)}")
    print("\nTroubleshooting steps:")
    print("1. Check if the Aura instance is running")
    print("2. Verify credentials are correct")
    print("3. Check network connectivity")
    exit(1)

print("\n" + "=" * 50)
print("Test completed.")