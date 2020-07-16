/**
         (                                      
   (     )\ )                                   
 ( )\   (()/(   (    ) (        (        (  (   
 )((_)   /(_)) ))\( /( )(   (   )\  (    )\))(  
((_)_   (_))  /((_)(_)|()\  )\ |(_) )\ )((_))\  
 / _ \  | |  (_))((_)_ ((_)_(_/((_)_(_/( (()(_) 
| (_) | | |__/ -_) _` | '_| ' \)) | ' \)) _` |  
 \__\_\ |____\___\__,_|_| |_||_||_|_||_|\__, |  
                                        |___/   

Refer to Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8. 3-4 (1992): 279-292
for a detailed discussion on Q Learning
*/
#include "CQLearningController.h"
#include <vector>
#include <iostream>

using namespace std;


CQLearningController::CQLearningController(HWND hwndMain):
	CDiscController(hwndMain),
	_grid_size_x(CParams::WindowWidth / CParams::iGridCellDim + 1),
	_grid_size_y(CParams::WindowHeight / CParams::iGridCellDim + 1)
{
}
/**
 The update method should allocate a Q table for each sweeper (this can
 be allocated in one shot - use an offset to store the tables one after the other)

 You can also use a boost multiarray if you wish

 Initialises a 2d vector of states with 4 action per state with values of 0.0
*/
void CQLearningController::InitializeLearningAlgorithm(void)
{
	for (uint i = 0; i < _grid_size_x; i++) {
		vector<vector<double> > temp;
		for (uint j = 0; j < _grid_size_y; j++) {
			vector<double> temp2(4, 0.0);
			temp.push_back(temp2);
		}
		Qmatrix.push_back(temp);
	}

	resetDeadQ();
}
/**
 The immediate reward function. This computes a reward upon achieving the goal state of
 collecting all the mines on the field. It may also penalize movement to encourage exploring all directions and 
 of course for hitting supermines/rocks!
*/
double CQLearningController::R(uint x,uint y, uint sweeper_no){
	
	double reward = 0.0;

	int found = ((m_vecSweepers[sweeper_no])->CheckForObject(m_vecObjects, CParams::dMineScale));

	if (found >= 0)
	{
		switch (m_vecObjects[found]->getType()) {
		case CDiscCollisionObject::Mine: {
			if (!m_vecObjects[found]->isDead()) {
				//If we hit a mine, return a nice positve reward
				reward = 100.0;
			}
			break;
		}
		case CDiscCollisionObject::Rock: {
			//we hit a rock and died, so return a negative reward
			reward = -100.0;
			break;
		}
		case CDiscCollisionObject::SuperMine: {
			//we hit a supermine and died, returna negative reward
			reward = -100.0;
			break;
		}
		}
	}

	return reward;
}

/**
The update method. Main loop body of our Q Learning implementation
See: Watkins, Christopher JCH, and Peter Dayan. "Q-learning." Machine learning 8. 3-4 (1992): 279-292
*/
bool CQLearningController::Update(void)
{		
	// Writes out data to a file to be used in the report

	if (m_iterations > 50) {

		ofstream out;
		out.open("output.txt");

		double tGathered = 0;
		double tDeaths = 0;
		int highestGathered = 0;

		for (int i = 0; i < 50; i++) {
			tGathered += m_vecMostMinesGathered[i];
			tDeaths += m_vecDeaths[i];
			if (m_vecAvMinesGathered[i] > highestGathered) highestGathered = m_vecAvMinesGathered[i];
		}

		out << "Most mines gathered: " << highestGathered << endl;
		out << "Average mines gathered: " << double(tGathered / 50) << endl;
		out << "Average mines deaths: " << double(tDeaths / 50) << endl;
		out.close();
	}


	//m_vecSweepers is the array of minesweepers
	//everything you need will be m_[something] ;)
	uint cDead = std::count_if(m_vecSweepers.begin(),
							   m_vecSweepers.end(),
						       [](CDiscMinesweeper * s)->bool{
								return s->isDead();
							   });
	if (cDead == CParams::iNumSweepers){
		printf("All dead ... skipping to next iteration\n");
		m_iTicks = CParams::iNumTicks;
		epsilon -= 0.01;
		learning_rate -= 0.02;
		resetDeadQ();
		m_iterations++;
	}

	for (uint sw = 0; sw < CParams::iNumSweepers; ++sw) {
		if (m_vecSweepers[sw]->isDead()) continue;

		//1:::Observe the current state:

		SVector2D<int> position = m_vecSweepers[sw]->Position();
		position /= CParams::iGridCellDim;

		//2:::Select action with highest historic return:
		
		// Epsilon Greedy Strategy was used here

		double r = RandFloat();

		// Exploration is chosen

		if (r < epsilon) {
			double highest = Qmatrix[position.x][position.y][0];

			for (uint i = 1; i < 4; i++) {
				if (highest < Qmatrix[position.x][position.y][i]) highest = Qmatrix[position.x][position.y][i];
			}

			vector<int> duplicates;

			for (uint i = 0; i < 4; i++) {
				if (highest == Qmatrix[position.x][position.y][i]) duplicates.push_back(i);
			}

			int randaction = RandInt(0, duplicates.size() - 1);
			m_vecSweepers[sw]->setRotation((ROTATION_DIRECTION)duplicates[randaction]);
		}
		// Exploitation is chosen
		else {
			double highest = Qmatrix[position.x][position.y][0];

			for (uint i = 1; i < 4; i++) {
				if (highest < Qmatrix[position.x][position.y][i]) highest = i;
			}

			int action = (int)highest;

			m_vecSweepers[sw]->setRotation((ROTATION_DIRECTION)action);
		}
	}
	//now call the parents update, so all the sweepers fulfill their chosen action

	CDiscController::Update(); //call the parent's class update. Do not delete this.
	
	for (uint sw = 0; sw < CParams::iNumSweepers; ++sw){
		if (m_vecSweepers[sw]->isDead() && dead_Q[sw]) continue;
		//TODO:compute your indexes.. it may also be necessary to keep track of the previous state
		// Allows a sweeper that has hit a mine to still run its iteration and update its reward

		else if (m_vecSweepers[sw]->isDead()) dead_Q[sw] = true;

		//3:::Observe new state:

		SVector2D<int> prev = m_vecSweepers[sw]->PrevPosition();
		prev /= CParams::iGridCellDim;

		SVector2D<int> current = m_vecSweepers[sw]->Position();
		current /= CParams::iGridCellDim;

		int action = (int)m_vecSweepers[sw]->getRotation();

		//4:::Update _Q_s_a accordingly:

		// Finds the action with the highest value in the specific position
		
		double highest = Qmatrix[current.x][current.y][0];

		for (uint i = 1; i < 4; i++) {
			if (highest < Qmatrix[current.x][current.y][i]) highest = Qmatrix[current.x][current.y][i];
		}
		
		Qmatrix[prev.x][prev.y][action] += learning_rate * (R(current.x, current.y, sw) + (discount_factor * highest) - Qmatrix[prev.x][prev.y][action]);

	}

	// Changes the values and resets the Q dead matrix every iteration

	if (m_iTicks == CParams::iNumTicks) {
		resetDeadQ();
		epsilon -= 0.01;
		learning_rate -= 0.02;
		m_iterations++;
	}

	return true;
}

/*
Initialises the dead_Q table to falses so that when it checks in the update function, it allows it to iterate and
update the Q value with a negative reward

*/

void CQLearningController::resetDeadQ() {
	dead_Q.clear();

	for (int i = 0; i < CParams::iNumSweepers; i++) {
		dead_Q.push_back(false);
	}
}


CQLearningController::~CQLearningController(void)
{
	//TODO: dealloc stuff here if you need to	
}
