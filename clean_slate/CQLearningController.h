#pragma once
#include "cdisccontroller.h"
#include "CParams.h"
#include "CDiscCollisionObject.h"
#include <cmath>
#include <vector>

typedef unsigned int uint;
class CQLearningController :
	public CDiscController
{
private:
	uint _grid_size_x;
	uint _grid_size_y;
	std::vector<std::vector<std::vector<double> > > Qmatrix;
	std::vector<bool> dead_Q;
	double discount_factor = 0.9; 
	double learning_rate = 0.9;
	double epsilon = 1.0;
	int m_iterations = 0;
	

public:
	CQLearningController(HWND hwndMain);
	virtual void InitializeLearningAlgorithm(void);
	double R(uint x, uint y, uint sweeper_no);
	virtual bool Update(void);
	virtual ~CQLearningController(void);	
	void resetDeadQ(); // Initialises the dead Q matrix that keeps count of which mines have been swept
};

