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
inline double inner_product (double* X1, double* X2, int dim)
{

	double result = 0.0;
	for (int i=0; i<dim; i++) result += X1[i]*X2[i];
	return result;

}

inline double c_factor (int N)
{

	double Nd = (double) N;
	double result;
	result = 2.0*(log(Nd-1.0)+EULER_CONSTANT) - 2.0*(Nd-1.0)/Nd;
	return result;

}

inline std::vector<int> sample_without_replacement (int k, int N, RANDOM_ENGINE& gen)
{

    /*
     * Sample k elements from the range [1, N] without replacement
     * k should be <= N
     * Source: https://www.gormanalysis.com/blog/random-numbers-in-cpp/
     */

    // Create an unordered set to store the samples
    std::unordered_set<int> samples;

    // Sample and insert values into samples
    for (int r=N-k+1; r<N+1; ++r)
    {
        int v = std::uniform_int_distribution<>(1, r)(gen);
        if (!samples.insert(v).second) samples.insert(r);
    }

    // Copy samples into vector
    std::vector<int> result(samples.begin(), samples.end());

    // Shuffle vector
    std::shuffle(result.begin(), result.end(), gen);

    return result;

}


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
//	double* X;	// unused in original code
	std::vector<double> normal_vector;
	std::vector<double> point;
	Node* left;
	Node* right;
	std::string node_type;

	Node (int, int, double*, double*, int, Node*, Node*, std::string);
	~Node ();

};

Node::Node (int size_in, int dim_in, double* normal_vector_in, double* point_in, int e_in, Node* left_in, Node* right_in, std::string node_type_in)
{

	e = e_in;
	size = size_in;
	for (int i=0; i<dim_in; i++)
	{
		normal_vector.push_back(normal_vector_in[i]);
		point.push_back(point_in[i]);
	}
	left = left_in;
	right = right_in;
	node_type = node_type_in;

}

Node::~Node ()
{

}


/****************************
        Class iTree
 ****************************/
class iTree
{

    private:
	int exlevel;
	int e;
	int size;
//	int* Q;		// unused in original code
	int dim;
	int limit;
	int exnodes;
	double* point;
	double* normal_vector;
//	double* X;	// unused in original code
    protected:

    public:
	Node* root;

	iTree ();
	~iTree ();
	void build_tree (double*, int, int, int, int, RANDOM_ENGINE&, int);
	Node* add_node (double*, int, int, RANDOM_ENGINE&);

};

iTree::iTree ()
{

}

iTree::~iTree ()
{

}

void iTree::build_tree (double* X_in, int size_in, int e_in, int limit_in, int dim_in, RANDOM_ENGINE& random_engine_in, int exlevel_in=0)
{

	exlevel = exlevel_in;
	e = e_in;
	size = size_in;
	dim = dim_in;
	limit = limit_in;
	exnodes = 0;
	point = new double [dim];
	normal_vector = new double [dim];
	root = add_node (X_in, size_in, e_in, random_engine_in);
	delete [] point;
	delete [] normal_vector;

}

Node* iTree::add_node (double* X_in, int size_in, int e_in, RANDOM_ENGINE& random_engine_in)
{

	e = e_in;
	if (e_in >= limit || size_in <= 1) {

		Node* left = NULL;
		Node* right = NULL;
		exnodes += 1;
		Node* node = new Node (size_in, dim, normal_vector, point, e_in, left, right, "exNode");
		return node;

	} else {

		/* Find mins, maxs */
		std::vector<double> Xmins, Xmaxs;
		for (int i=0; i<dim; i++)
		{
			Xmins.push_back(X_in[i]);
			Xmaxs.push_back(X_in[i]);
			for (int j=1; j<size_in; j++)
			{
				int index = i+j*dim;
				if (Xmins[i] > X_in[index]) Xmins[i] = X_in[index];
				if (Xmaxs[i] < X_in[index]) Xmaxs[i] = X_in[index];
			}
		}

		/* Pick a random point on splitting hyperplane */
		for (int i=0; i<dim; i++)
			point[i] = std::uniform_real_distribution<double> (Xmins[i], Xmaxs[i])(random_engine_in);

		/* Pick a random normal vector according to specified extension level */
		for (int i=0; i<dim; i++)
			normal_vector[i] = std::normal_distribution<double> (0.0, 1.0)(random_engine_in);
		std::vector<int> normvect_zero_index = sample_without_replacement (dim-exlevel-1, dim, random_engine_in);
		for (int j=0; j<dim-exlevel-1; j++)
			normal_vector[normvect_zero_index[j]-1] = 0.0;

		/* Implement splitting criterion */
		double innerprod, pdotn;
		std::vector<double> XL, XR;
		int sizeXL = 0, sizeXR = 0;

		pdotn = inner_product (point, normal_vector, dim);
		for (int i=0; i<size_in; i++)
		{
			int index = i*dim;
			innerprod = inner_product (&X_in[index], normal_vector, dim);
			if (innerprod < pdotn) {
				for (int j=0; j<dim; j++) XL.push_back(X_in[j+index]);
				sizeXL += 1;
			} else {
				for (int j=0; j<dim; j++) XR.push_back(X_in[j+index]);
				sizeXR += 1;
			}
		}

		Node* left = add_node (&XL[0], sizeXL, e_in+1, random_engine_in);
		Node* right = add_node (&XR[0], sizeXR, e_in+1, random_engine_in);

		Node* node = new Node (size_in, dim, normal_vector, point, e_in, left, right, "inNode");
		return node;

	}

}


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

Path::Path (int dim_in, double* x_in, iTree itree_in)
{

	dim = dim_in;
	x = x_in;
	e = 0.0;
	pathlength = find_path (itree_in.root);

}

Path::~Path ()
{

}

double Path::find_path (Node* node_in)
{

	if (node_in[0].node_type == "exNode") {

		if (node_in[0].size <= 1) {
			return e;
		} else {
			e = e + c_factor (node_in[0].size);
			return e;
		}

	} else {

		e += 1.0;

		double xdotn, pdotn, plength;
		pdotn = inner_product (&node_in[0].point[0], &node_in[0].normal_vector[0], dim);
		xdotn = inner_product (x, &node_in[0].normal_vector[0], dim);
		if (xdotn < pdotn) {
			path_list.push_back('L');
			plength = find_path (node_in[0].left);
		} else {
			path_list.push_back('R');
			plength = find_path (node_in[0].right);
		}
		return plength;

	}

}


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

iForest::iForest (int dim_in, int ntrees_in, int sample_in, int limit_in=0, int exlevel_in=0)
{

	ntrees = ntrees_in;
	dim = dim_in;
	sample = sample_in;
	limit = limit_in;
	if (limit_in == 0) limit = (int) ceil(log2(sample));
	exlevel = exlevel_in;
	CheckExtensionLevel();
	c = c_factor (sample);
	Trees = new iTree [ntrees];
	RANDOM_SEED_GENERATOR random_seed_generator;
	random_seed = random_seed_generator();

}

iForest::~iForest ()
{

	delete [] Trees;

}

void iForest::CheckExtensionLevel ()
{

	if (exlevel < 0)
		std::cout << "Extension level must be an integer between 0 and " << dim-1 << "." << std::endl;
	if (exlevel > dim-1)
		std::cout << "Your data has " << dim << " dimensions. Extension level cannot be higher than " << dim-1 << "." << std::endl;

}

void iForest::fit (double* X_in, int nobjs_in)
{
	X = X_in;
	nobjs = nobjs_in;
	std::vector<double> Xsubset;

	for (int i=0; i<ntrees; i++)
	{
		/* Select a random subset of X_in of size sample_in */
		RANDOM_ENGINE random_engine (random_seed+i);
		std::vector<int> sample_index = sample_without_replacement (sample, nobjs, random_engine);
		Xsubset.clear();
		for (int j=0; j<sample; j++) Xsubset.push_back(X[sample_index[j]-1]);

		Trees[i].build_tree (&Xsubset[0], sample, 0, limit, dim, random_engine, exlevel);
	}

}

void iForest::predict (double* S, double* X_in=NULL, int size_in=0)
{

	if (X_in == NULL)
	{
		X_in = X;
		size_in = nobjs;
	}

	double htemp, havg;
	for (int i=0; i<size_in; i++)
	{
		htemp = 0.0;
		for (int j=0; j<ntrees; j++)
		{
			Path path (dim, &X_in[i*dim], Trees[j]);
			htemp += path.pathlength;
		}
		havg = htemp/ntrees;
		S[i] = std::pow(2.0, -havg/c);
	}

}
