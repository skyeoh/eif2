#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>

#define EULER_CONSTANT 0.5772156649

#define RANDOM_ENGINE std::mt19937_64
#define RANDOM_SEED_GENERATOR std::random_device


/********************************
        Utility functions
 ********************************/
inline double inner_product (double*, double*, int);
inline double c_factor (int);
inline std::vector<int> sample_without_replacement (int, int, RANDOM_ENGINE&);


/****************************
        Class Node
 ****************************/
class Node
{

    private:
        int e;
    protected:

    public:
        int size;
//      double* X;      // unused in original code
        std::vector<double> normal_vector;
        std::vector<double> point;
        Node* left;
        Node* right;
        std::string node_type;

        Node (int, int, double*, double*, int, Node*, Node*, std::string);
        ~Node ();

};


/****************************
        Class iTree
 ****************************/
class iTree
{

    private:
        int exlevel;
        int e;
        int size;
//      int* Q;         // unused in original code
        int dim;
        int limit;
        int exnodes;
        double* point;
        double* normal_vector;
//      double* X;      // unused in original code
    protected:

    public:
        Node* root;

        iTree ();
        ~iTree ();
        void build_tree (double*, int, int, int, int, RANDOM_ENGINE&, int);
        Node* add_node (double*, int, int, RANDOM_ENGINE&);

};


/*************************
        Class Path
 *************************/
class Path
{

    private:
        std::vector<char> path_list;
        int dim;
        double* x;
        double e;
    protected:

    public:
        double pathlength;

        Path (int, double*, iTree);
        ~Path ();
        double find_path (Node*);

};


/****************************
        Class iForest
 ****************************/
class iForest
{

    private:
        int ntrees;
        int nobjs;
        int dim;
        int sample;
        int limit;
        int exlevel;
        double* X;
        double c;
        iTree* Trees;
        unsigned random_seed;
    protected:

    public:
        iForest (int, int, int, int, int);
        ~iForest ();
        void CheckExtensionLevel ();
        void fit (double*, int);
        void predict (double*, double*, int);

};
