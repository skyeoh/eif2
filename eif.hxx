#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>
#if ENABLE_OPENMP
#include <omp.h>
#endif

#define EULER_CONSTANT 0.5772156649

#define RANDOM_ENGINE std::mt19937_64
#define RANDOM_SEED_GENERATOR std::random_device


/****************************
        Class Node
 ****************************/
class Node
{

       /*
	* A Node object
	* This object represents a single node on a tree (iTree object).
	* It contains information on the hyperplanes used for splitting data at the node,
	* on whether the node is external or internal, on the child nodes, etc.
	*
	* Attributes
	* ----------
	* e : int
	*     Depth of the node on the tree
	* size : int
	*     Size of dataset present at the node
	* normal_vector : vector of doubles
	*     Normal vector used to build the hyperplane that splits the data at the node
	* point : vector of doubles
	*     Intercept point through which the hyperplane passes
	* left : Node*
	*     Left child node
	* right : Node*
	*     Right child node
	* node_type : string object
	*     Type of the node: external (exNode) or internal (inNode)
	*
	* Methods
	* -------
	* Node (int, int, double*, double*, int, Node*, Node*, std::string)
	*     Initialize a Node object (constructor)
	*     This populates the members of the Node object.
	* ~Node ()
	*     Destroy a Node object (destructor)
	*/

    private:

    protected:

    public:
	int e;
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

       /*
	* An iTree object
	* It represents a single tree in the forest. It is built (trained)
	* using a unique subsample.
	*
	* Attributes
	* ----------
	* exlevel : int
	*     Extension level used in splitting criterion
	* e : int
	*     Depth of the tree
	* size : int
	*     Size of sample used to create the tree
	* dim : int
	*     Dimension of the dataset
	* limit : int
	*     Maximum depth the tree can reach before creation is terminated
	* exnodes : int
	*     Number of external nodes in the tree
	* root : Node*
	*     Root node of the tree
	*     It is the first node from which the tree is created and
	*     all the other nodes originate.
	*
	* Methods
	* -------
	* iTree ()
	*     Initialize an iTree object (constructor)
	* ~iTree ()
	*     Destroy an iTree object (destructor)
	* void build_tree (double*, int, int, int, int, RANDOM_ENGINE&, int)
	*     Build (train) a tree
	* Node* add_node (double*, int, int, RANDOM_ENGINE&)
	*     Add a node to the tree
        *     The tree is built recursively from the node.
	*/

    private:
        int exlevel;
        int e;
        int size;
//      int* Q;         // unused in original code
        int dim;
        int limit;
        int exnodes;
//	double* point;		// in original code, but not necessary
//	double* normal_vector;	// in original code, but not necessary
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

       /*
	* A Path object
	* Given a single tree (iTree) and a single data point,
	* this object computes the length of the path traversed by the data point
	* on the tree when it reaches an external node.
	*
	* Attributes
	* ----------
	* dim : int
	*     Dimension of the data point
	*     This must be equal to the dimension of the training dataset.
	* x : double*
	*     Pointer to the data point
	* e : double
	*     Length of the path traversed by the data point on the tree
	* path_list : vector of chars
	*     Path traversed by the data point along the tree, e.g. "RLRLRRLL"
	* pathlength : double
	*     Length of the path traversed by the data point on the tree
	*
	* Methods
	* -------
	* Path (int, double*, iTree)
	*     Initialize a Path object (constructor)
	* ~Path ()
	*     Destroy a Path object (destructor)
	* double find_path (Node*)
	*    Find the path a data point takes along a tree
	*    and compute the length of the path
	*/

    private:
        int dim;
        double* x;
        double e;
    protected:

    public:
        std::vector<char> path_list;
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

       /*
        * An iForest object
	* This object holds the data and the trained trees (iTree objects).
	*
	* Attributes
	* ----------
	* ntrees : int
	*     Number of trees to train
	* nobjs : int
	*     Size of dataset
	* dim : int
	*     Dimension of dataset
	* sample : int
	*     Size of sample to use for tree creation
	* limit : int
	*     Maximum depth a tree can have
	* exlevel : int
	*     Extension level to use in splitting criterion
	* X : double*
	*     Pointer to dataset to use for training
	* c : double
	*     Multiplicative factor used in computing anomaly scores
	* Trees : iTree*
	*     Pointer to trees to train
	* random_seed : unsigned
	*     Seed for random number generation
	*
	* Methods
	* -------
	* bool CheckExtensionLevel ()
	*     Check the validity of the extension level provided
	*     based on the dimension of the dataset used for training
	* bool CheckSampleSize ()
	*     Check the validity of the sample size provided
	*     based on the size of the dataset used for training
	* iForest (int, int, int, int, int)
	*     Initialize an iForest object (constructor)
	* ~iForest ()
	*     Destroy an iForest object (destructor)
	*     This deallocates all memory used in creating the trees.
	* void fit (double*, int, int)
	*     Build and train forest of trees
	* void predict (double*, double*, int)
	*     Compute anomaly scores for input dataset
	*     Input dataset could be different from training dataset.
	* void OutputTreeNodes (int)
	*     Output properties of all nodes for a tree
	*/

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

	bool CheckExtensionLevel ();
	bool CheckSampleSize ();
    protected:

    public:
        iForest (int, int, int, int, int);
        ~iForest ();
        void fit (double*, int, int);
        void predict (double*, double*, int);
	void OutputTreeNodes (int);

};


/********************************
        Utility functions
 ********************************/
inline double inner_product (double*, double*, int);
inline double c_factor (int);
inline std::vector<int> sample_without_replacement (int, int, RANDOM_ENGINE&);
void output_tree_node (Node*, std::string);
void delete_tree_node (Node*);
