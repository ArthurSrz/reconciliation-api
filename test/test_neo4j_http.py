#!/usr/bin/env python3
"""
Test alternative methods to connect to Neo4j Aura when Bolt protocol is blocked.
Uses HTTP/HTTPS APIs instead of Bolt (port 7687).
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
print("Testing Neo4j Aura HTTP/HTTPS Connection Methods")
print("=" * 70)
print(f"Instance: {AURA_INSTANCE}")
print(f"Username: {NEO4J_USER}")
print(f"Database: {NEO4J_DATABASE}\n")

# Test different HTTP endpoints
test_methods = [
    {
        "name": "Neo4j HTTP API (HTTPS on default port)",
        "url": f"https://{AURA_INSTANCE}/db/{NEO4J_DATABASE}/tx/commit",
        "method": "POST",
        "headers": {
            "Authorization": f"Basic {base64_auth}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        "data": {
            "statements": [
                {
                    "statement": "RETURN 1 as test, 'Hello from HTTP!' as message"
                }
            ]
        }
    },
    {
        "name": "Neo4j Browser Endpoint",
        "url": f"https://{AURA_INSTANCE}/browser/",
        "method": "GET",
        "headers": {
            "Authorization": f"Basic {base64_auth}"
        }
    },
    {
        "name": "Neo4j Discovery API",
        "url": f"https://{AURA_INSTANCE}/",
        "method": "GET",
        "headers": {
            "Authorization": f"Basic {base64_auth}",
            "Accept": "application/json"
        }
    },
    {
        "name": "Neo4j Cypher Endpoint (Transactional)",
        "url": f"https://{AURA_INSTANCE}/db/data/transaction/commit",
        "method": "POST",
        "headers": {
            "Authorization": f"Basic {base64_auth}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        "data": {
            "statements": [
                {
                    "statement": "RETURN 1 as test"
                }
            ]
        }
    }
]

successful_method = None

for test in test_methods:
    print(f"\n{'=' * 70}")
    print(f"Testing: {test['name']}")
    print(f"URL: {test['url']}")
    print(f"Method: {test['method']}")
    print("-" * 70)

    try:
        if test['method'] == 'GET':
            response = httpx.get(
                test['url'],
                headers=test['headers'],
                timeout=10.0,
                follow_redirects=True
            )
        else:  # POST
            response = httpx.post(
                test['url'],
                headers=test['headers'],
                json=test.get('data'),
                timeout=10.0,
                follow_redirects=True
            )

        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")

        if response.status_code in [200, 201]:
            print("✅ SUCCESS!")
            print(f"Response preview: {response.text[:500]}")

            # Try to parse JSON
            try:
                json_response = response.json()
                print(f"\nJSON Response:")
                print(json.dumps(json_response, indent=2)[:1000])

                # Check if we got data back
                if 'results' in json_response:
                    print("\n🎉 CYPHER QUERY WORKED!")
                    print("This method can be used to query Neo4j!")
                    successful_method = test
                    break
            except:
                pass
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"Response: {response.text[:300]}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

print("\n" + "=" * 70)

if successful_method:
    print("\n🎉 SUCCESS! Found working method:")
    print(f"   {successful_method['name']}")
    print(f"   URL: {successful_method['url']}")
    print("\nThis endpoint can be used to bypass Bolt protocol!")
else:
    print("\n⚠️  No HTTP method worked. Possible reasons:")
    print("   - Neo4j Aura HTTP API might not be exposed")
    print("   - Firewall blocking HTTP access")
    print("   - Authentication issues")
    print("\n   Try checking: https://console.neo4j.io")

print("=" * 70)
