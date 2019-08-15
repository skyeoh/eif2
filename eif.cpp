#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <optional>

#define EULER_CONSTANT 0.5772156649


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


/****************************
        Class Node
 ****************************/
class Node
{

    private:
	int e;
	int size;
//	double* X;	// unused in original code
	std::vector<double> normal_vector;
	std::vector<double> point;
	Node left;
	Node right;
	std::string node_type;
    public:
	Node (int, int, double*, double*, int, Node, Node, std::string);
	~Node ();

};

Node::Node (int size_in, int dim_in, double* normal_vector_in, double* point_in, int e_in, Node left_in, Node right_in, std::string node_type_in)
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
	Node root;
    public:
	iTree (double*, int, int, int, int, int);
	~iTree ();
	Node make_tree (double*, int, int);

};

iTree::iTree (double* X_in, int size_in, int e_in, int limit_in, int dim_in, int exlevel_in=0)
{

	exlevel = exlevel_in;
	e = e_in;
	size = size_in;
	dim = dim_in;
	limit = limit_in;
	exnodes = 0;
	point = new double [dim];
	normal_vector = new double [dim];
	root = make_tree (X_in, size_in, e_in);

}

iTree::~iTree ()
{

	delete [] point;
	delete [] normal_vector;

}

Node iTree::make_tree (double* X_in, int size_in, int e_in)
{

	e = e_in;
	if (e_in >= limit || size_in <= 1) {

		std::optional<Node> left;
		std::optional<Node> right;
		exnodes += 1;
		Node node (size_in, dim, normal_vector, point, e_in, left, right, "exNode");
		return node;

	} else {

		/* Find min, max */
		std::vector<double> mins, maxs;
		for (int i=0; i<dim; i++)
		{
			mins.push_back(X_in[i]);
			maxs.push_back(X_in[i]);
			for (int j=1; j<size_in; j++)
			{
				int index = i+j*dim;
				if (mins[i] > X_in[index]) mins[i] = X_in[index];
				if (maxs[i] < X_in[index]) maxs[i] = X_in[index];
			}
		}

		/* Insert code that picks a random point on splitting hyperplane */

		/* Insert code that picks random normal vector according to specified extension level */

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

		Node left = make_tree (&XL[0], sizeXL, e_in+1);
		Node right = make_tree (&XR[0], sizeXR, e_in+1);

		Node node (size_in, dim, normal_vector, point, e_in, left, right, "inNode");
		return node;

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
    public:
	iForest (double*, int, int, int, int, int, int);
	~iForest ();
	void CheckExtensionLevel ();
	void fit ();
	void predict ();

};

iForest::iForest (double* X_in, int nobjs_in, int dim_in, int ntrees_in, int sample_in, int limit_in=0, int exlevel_in=0)
{

	X = X_in;
	ntrees = ntrees_in;
	nobjs = nobjs_in;
	dim = dim_in;
	sample = sample_in;
	limit = limit_in;
	if (limit_in == 0) limit = (int) ceil(log2(sample));
	exlevel = exlevel_in;
	CheckExtensionLevel();
	c = c_factor (sample);
	Trees = new iTree [ntrees];

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

void iForest::fit ()
{

	int i;
	double* Xsubset;

	for (i=0; i<ntrees; i++)
	{
		/* Insert code that selects a random subset of X_in of size sample_in */
		Trees[i](Xsubset, sample, 0, limit, dim, exlevel);
	}

}

void iForest::predict ()
{

}
