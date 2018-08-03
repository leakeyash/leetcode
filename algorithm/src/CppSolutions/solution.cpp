#include <vector>
#include <map>
#include<iostream>
using namespace std;

class Solution {
public:
	void testTwoSum() {
		vector<int> nums;
		nums.push_back(2);
		nums.push_back(7);
		nums.push_back(11);
		nums.push_back(15);
		int target = 9;
		vector<int> results = twoSum(nums, target);
		cout << results[0] <<", "<<results[1] << endl;
	}
	vector<int> twoSum(vector<int>& nums, int target) {
		map<int, int> hash;
		vector<int> result;
		for (int i = 0; i < nums.size(); i++) {
			int temp = target - nums[i];
			if (hash.find(temp) != hash.end()) {
				result.push_back(hash[temp] );
				result.push_back(i);
				return result;
			}
			hash[nums[i]] = i;
		}
		return result;
	}
};