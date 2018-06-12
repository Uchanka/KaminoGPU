# pragma once

# include "../include/KaminoHeader.cuh"

class KaminoTimer
{
private:
	cudaEvent_t start;
	cudaEvent_t stop;
	float timeElapsed;
public:
	KaminoTimer();
	~KaminoTimer();

	void startTimer();
	float stopTimer();
};
