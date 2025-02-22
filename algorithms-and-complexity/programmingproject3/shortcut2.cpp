#include <iostream>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

const long long INF = 1e9;

struct Edge {
    int to;
    long long weight;
};

struct State {
    long long distance;
    int node;
    int skips_used;

    // Overload the < operator for priority_queue (min-heap behavior)
    bool operator<(const State& other) const {
        return distance > other.distance;  // Reverse for min-heap
    }
};

int N, M;
int S, T;
long long B;
vector<vector<Edge> > graph;

// Dijkstra's algorithm for layered graph with up to 'k' skips
bool isPossible(int k) {
    // Distance array: dist[node][skips_used]
    vector<vector<long long> > dist(N + 1, vector<long long>(k + 2, INF));
    priority_queue<State> pq;

    // Start from source S with 0 skips used
    dist[S][0] = 0;
    pq.push((State){0, S, 0});

    while (!pq.empty()) {
        State current = pq.top();
        pq.pop();

        long long cur_dist = current.distance;
        int u = current.node;
        int skips_used = current.skips_used;

        // Early stopping if we reached destination and within time limit
        if (u == T && cur_dist <= B) {
            return true;
        }

        // Skip processing if we've already found a better path
        if (cur_dist > dist[u][skips_used]) continue;

        // Traverse all edges from node 'u'
        for (size_t i = 0; i < graph[u].size(); ++i) {
            int v = graph[u][i].to;
            long long weight = graph[u][i].weight;

            // 1. Move without skipping the edge
            if (cur_dist + weight < dist[v][skips_used]) {
                dist[v][skips_used] = cur_dist + weight;
                pq.push((State){dist[v][skips_used], v, skips_used});
            }

            // 2. Skip this edge if we haven't used all skips
            if (skips_used < k && cur_dist < dist[v][skips_used + 1]) {
                dist[v][skips_used + 1] = cur_dist;
                pq.push((State){dist[v][skips_used + 1], v, skips_used + 1});
            }
        }
    }

    // After processing, check if we can reach T with any number of skips within B time
    for (int skips = 0; skips <= k; ++skips) {
        if (dist[T][skips] <= B) {
            return true;
        }
    }

    return false;
}

// Binary search to find minimum k
int findMinimumK() {
    int left = 0, right = N - 1;
    int answer = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (isPossible(mid)) {
            answer = mid;        // Found a valid k, try to find a smaller one
            right = mid - 1;
        } else {
            left = mid + 1;      // Need more skips
        }
    }

    return answer;
}

int main() {
    // Input
    cin >> N >> M >> S >> T >> B;

    graph.resize(N + 1);

    for (int i = 0; i < M; ++i) {
        int u, v;
        long long w;
        cin >> u >> v >> w;
        graph[u].push_back((Edge){v, w});
    }

    int result = findMinimumK();
    cout << result << endl;

    return 0;
}
