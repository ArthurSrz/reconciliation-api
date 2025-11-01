#!/usr/bin/env python3
"""
Test Neo4j Aura Query API (HTTP-based alternative to Bolt protocol)
This is the official Neo4j HTTP API available on Aura since 2024.
URL format: https://HOSTNAME/db/DATABASE/query/v2
"""

import httpx
import base64
import json
from dotenv import load_dotenv
import os

# Load credentials
load_dotenv()

NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')
AURA_INSTANCE = "f768707e.databases.neo4j.io"

# Create basic auth header
auth_string = f"{NEO4J_USER}:{NEO4J_PASSWORD}"
auth_bytes = auth_string.encode('ascii')
base64_auth = base64.b64encode(auth_bytes).decode('ascii')

print("=" * 70)
print("Testing Neo4j Aura Query API (Official HTTP API)")
print("=" * 70)
print(f"Instance: {AURA_INSTANCE}")
print(f"Username: {NEO4J_USER}")
print(f"Database: {NEO4J_DATABASE}\n")

# Query API endpoint
query_api_url = f"https://{AURA_INSTANCE}/db/{NEO4J_DATABASE}/query/v2"

print(f"Query API URL: {query_api_url}\n")

# Test queries
test_queries = [
    {
        "name": "Simple RETURN test",
        "query": "RETURN 1 as number, 'Hello from Query API!' as message"
    },
    {
        "name": "Database info",
        "query": "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
    },
    {
        "name": "Count all nodes",
        "query": "MATCH (n) RETURN count(n) as total_nodes"
    },
    {
        "name": "Search for 'promesse' or 'aube'",
        "query": """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS 'promesse'
               OR toLower(n.name) CONTAINS 'aube'
               OR toLower(n.title) CONTAINS 'promesse'
               OR toLower(n.title) CONTAINS 'aube'
            RETURN n
            LIMIT 10
        """
    }
]

successful = False

for test in test_queries:
    print(f"\n{'=' * 70}")
    print(f"Test: {test['name']}")
    print(f"Query: {test['query'][:80]}...")
    print("-" * 70)

    try:
        # Prepare request
        headers = {
            "Authorization": f"Basic {base64_auth}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        payload = {
            "statement": test['query']
        }

        # Make request
        response = httpx.post(
            query_api_url,
            headers=headers,
            json=payload,
            timeout=15.0
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("✅ SUCCESS!")
            successful = True

            try:
                result = response.json()
                print(f"\nResponse structure:")
                print(f"  Keys: {list(result.keys())}")

                if 'data' in result:
                    print(f"  Data: {json.dumps(result['data'], indent=2)[:500]}")
                elif 'values' in result:
                    print(f"  Values: {json.dumps(result['values'], indent=2)[:500]}")
                else:
                    print(f"  Full response: {json.dumps(result, indent=2)[:800]}")

                # If this is the search query and we got results
                if 'promesse' in test['query'].lower() and result:
                    print("\n🎉 FOUND DATA ABOUT 'LA PROMESSE DE L'AUBE'!")
                    print("The database contains relevant data!")

            except Exception as e:
                print(f"Could not parse JSON: {e}")
                print(f"Raw response: {response.text[:500]}")

        elif response.status_code == 401:
            print(f"❌ Authentication failed")
            print(f"Response: {response.text}")
        elif response.status_code == 403:
            print(f"❌ Access denied - Query API might not be enabled")
            print(f"Response: {response.text}")
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text[:300]}")

        # If first query works, continue with others
        if not successful:
            print("\n⚠️  Query API not working, skipping remaining tests...")
            break

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        break

print("\n" + "=" * 70)

if successful:
    print("\n🎉 SUCCESS! Neo4j Query API is working!")
    print(f"\n✅ Use this endpoint: {query_api_url}")
    print("✅ This can replace the Bolt driver in your application!")
    print("\nNext steps:")
    print("  1. Update reconciliation_api.py to use Query API")
    print("  2. Query about 'La Promesse de l'aube'")
else:
    print("\n⚠️  Query API not accessible. Possible reasons:")
    print("   - Query API might not be enabled on this Aura instance")
    print("   - Instance might need to be restarted")
    print("   - Check: https://console.neo4j.io")

print("=" * 70)
