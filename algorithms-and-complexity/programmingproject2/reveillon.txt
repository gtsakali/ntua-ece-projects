#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

long long maximize_happiness(int N, int K, const vector<int>& happiness) {
    vector<long long> dp(N + 1, 0);

    for (int i = 1; i <= N; ++i) {
        int max_happiness = 0;
        for (int j = 1; j <= min(K, i); ++j) {
            max_happiness = max(max_happiness, happiness[i - j]);
            dp[i] = max(dp[i], dp[i - j] + j * (long long)max_happiness);
        }
    }

    return dp[N];
}

int main() {
    int N, K;
    cin >> N >> K;
    vector<int> happiness(N);

    for (int i = 0; i < N; ++i) {
        cin >> happiness[i];
    }

    cout << maximize_happiness(N, K, happiness) << endl;

    return 0;
}