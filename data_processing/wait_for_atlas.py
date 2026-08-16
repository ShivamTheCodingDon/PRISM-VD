import time
import sys
import logging
from atlas_adapter import generate_atlas_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("atlas_wait")

# Complex sample code to test parsing readiness
COMPLEX_SAMPLE = """
#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node* next;
};

void complex_logic(struct Node* head, int val) {
    if (head == NULL) return;
    
    struct Node* curr = head;
    while (curr->next != NULL) {
        if (curr->data == val) {
            printf("Found match: %d\\n", curr->data);
            for (int i=0; i<10; i++) {
                curr->data += i;
            }
        } else {
            curr = curr->next;
        }
    }
    
    switch (val) {
        case 1: head->data = 100; break;
        case 2: head->data = 200; break;
        default: head->data = 0; break;
    }
}
"""

def wait_for_atlas(timeout_sec=600, poll_interval=10):
    start_time = time.time()
    logger.info("Starting active poll for ATLAS backend readiness...")
    
    while True:
        try:
            # Attempt to generate a graph from the complex sample
            g, _ = generate_atlas_graph(COMPLEX_SAMPLE, lang="c")

            
            # If the graph has nodes beyond just the root/fallback, ATLAS is ready
            node_count = len(g.nodes())
            if node_count > 5:
                logger.info(f"ATLAS is READY! Found {node_count} nodes. Proceeding immediately.")
                return True
            else:
                logger.info(f"ATLAS responding but graph is small ({node_count} nodes). Still warming up...")
        except Exception as e:
            logger.info(f"ATLAS backend not yet responsive. Retrying in {poll_interval}s...")
            
        if time.time() - start_time > timeout_sec:
            logger.error(f"ATLAS failed to initialize within {timeout_sec} seconds. Aborting.")
            return False
            
        time.sleep(poll_interval)

if __name__ == "__main__":
    if wait_for_atlas():
        sys.exit(0)
    else:
        sys.exit(1)
