from neo4j import GraphDatabase
import os


class Neo4jClient:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "Amit@123")
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print(f"Successfully connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            return False

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            print("Connection closed.")

    def run_query(self, query: str, parameters: dict = None):
        """Execute a Cypher query and return results."""
        if not self.driver:
            raise Exception("Not connected to database. Call connect() first.")
        
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test connection
    client = Neo4jClient()
    
    if client.connect():
        # Example: Get all node labels
        try:
            labels = client.run_query("CALL db.labels()")
            print(f"Available labels: {labels}")
            
            # Example: Count all nodes
            count = client.run_query("MATCH (n) RETURN count(n) as node_count")
            print(f"Total nodes: {count}")
        except Exception as e:
            print(f"Query error: {e}")
        finally:
            client.close()
