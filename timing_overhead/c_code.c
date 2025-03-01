#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_ARRAY_SIZE 1000
#define MAX_DEPTH 20
#define MAX_CHILDREN 1000

typedef struct PSTNode {
    struct PSTNode* children[MAX_CHILDREN];
    int count;
} PSTNode;

typedef struct ProbabilisticSuffixTree {
    PSTNode* root;
    int max_depth;
} ProbabilisticSuffixTree;

PSTNode* create_node() {
    PSTNode* node = (PSTNode*)malloc(sizeof(PSTNode));
    memset(node->children, 0, sizeof(node->children));
    node->count = 0;
    return node;
}

ProbabilisticSuffixTree* create_pst(int max_depth) {
    ProbabilisticSuffixTree* pst = (ProbabilisticSuffixTree*)malloc(sizeof(ProbabilisticSuffixTree));
    pst->root = create_node();
    pst->max_depth = max_depth;
    return pst;
}

void add_suffix(PSTNode* node, const int* suffix, int length) {
    for (int i = 0; i < length; i++) {
        int symbol = suffix[i];
        if (node->children[symbol] == NULL) {
            node->children[symbol] = create_node();
        }
        node = node->children[symbol];
    }
    node->count += 1;
}

void add_sequence(ProbabilisticSuffixTree* pst, const int* sequence, int length) {
    for (int i = 0; i < length; i++) {
        for (int j = i + 1; j <= i + pst->max_depth && j <= length; j++) {
            int suffix[j - i];
            memcpy(suffix, &sequence[i], (j - i) * sizeof(int));
            add_suffix(pst->root, suffix, j - i);
        }
    }
}

int predict_next(ProbabilisticSuffixTree* pst, const int* last_sequence, int length) {
    for (int i = length; i > 0; i--) {
        PSTNode* node = pst->root;
        for (int j = length - i; j < length; j++) {
            int symbol = last_sequence[j];
            if (node->children[symbol]) {
                node = node->children[symbol];
            } else {
                break;
            }
        }
        // Collect predictions from children nodes
        int total_count = 0;
        int predictions[MAX_CHILDREN] = {0};
        for (int k = 0; k < MAX_CHILDREN; k++) {
            if (node->children[k]) {
                predictions[k] = node->children[k]->count;
                total_count += predictions[k];
            }
        }

        // If there are predictions, sample one
        if (total_count > 0) {
            int rand_choice = rand() % total_count;
            for (int k = 0; k < MAX_CHILDREN; k++) {
                if (predictions[k] > 0) {
                    rand_choice -= predictions[k];
                    if (rand_choice < 0) {
                        // Ensure the predicted symbol is in the original sequence
                        for (int m = 0; m < length; m++) {
                            if (last_sequence[m] == k) {
                                return k; // predicted symbol
                            }
                        }
                    }
                }
            }
        }
    }
    return -1; // no prediction available
}

// Function to calculate GCD of two numbers
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

// Function to calculate LCM of two numbers
int lcm(int a, int b) {
    return abs(a * b) / gcd(a, b);
}

// Function to calculate hyperperiod (LCM of all periods in an array)
int calculate_hyperperiod(int* periods, int count) {
    int result = periods[0];
    for (int i = 1; i < count; i++) {
        result = lcm(result, periods[i]);
    }
    return result;
}

// Main function for demonstration purposes
int main() {
    srand(time(NULL)); // Seed for randomness

    FILE *fp;
    char path[1035];
    int array[MAX_ARRAY_SIZE];
    int size = 0;

    // Call the Python script and get the output
    fp = popen("python3 /home/pi/Desktop/date/c_linked.py", "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to run command\n");
        exit(1);
    }

    // Read the output a line at a time
    if (fgets(path, sizeof(path)-1, fp) != NULL) {
        char *token = strtok(path, "[, ]");
        while (token != NULL) {
            array[size++] = atoi(token);
            token = strtok(NULL, "[, ]");
        }
    }

    pclose(fp);

    int sequence[MAX_ARRAY_SIZE];
    for (int i = 0; i < size; i++) {
        sequence[i] = array[i];
        //printf("%d ", sequence[i]);    
    }
    //printf("\n");

    // Timing the creation of the probabilistic suffix tree
    clock_t start_time = clock();
    ProbabilisticSuffixTree* pst = create_pst(MAX_DEPTH);
    add_sequence(pst, sequence, size); // Add the entire sequence to the tree
    clock_t end_time = clock();
    double build_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("%f", build_time);

    // Timing the prediction
    int prediction_length = 5; // Change this to the length of the sequence you want to predict
    int last_sequence[MAX_DEPTH]; // Prepare the last sequence for prediction
    memcpy(last_sequence, &sequence[size - prediction_length], prediction_length * sizeof(int));

    start_time = clock();
    int predicted_symbol = predict_next(pst, last_sequence, prediction_length);
    end_time = clock();
    double predict_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    if (predicted_symbol != -1) {
        //printf("Predicted symbol: %d\n", predicted_symbol);
	int x=2;
    } else {
        //printf("No prediction available\n");
	int x=1;
    }
    printf("%f\n", predict_time);

    // Free allocated memory (you should implement a cleanup function for this)
    
    return 0;
}
