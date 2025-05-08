
#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>

using namespace std;

class Graph
{
    int V;
    vector<vector<int>> adjList;

public:
    Graph(int V)
    {
        this->V = V;
        adjList.resize(V);
    }

    void addEdge(int src, int dest)
    {
        adjList[src].push_back(dest);
        adjList[dest].push_back(src); // For undirected graph
    }

    vector<int> getNeighbors(int vertex)
    {
        return adjList[vertex];
    }
};

void parallelBFS(Graph &graph, int source, vector<bool> &visited, vector<int> &bfs_order)
{
    queue<int> q;
    q.push(source);
    visited[source] = true;

    while (!q.empty())
    {
        int current = q.front();
        q.pop();
        bfs_order.push_back(current);

        vector<int> neighbors = graph.getNeighbors(current);
#pragma omp parallel for
        for (int i = 0; i < neighbors.size(); ++i)
        {
            int neighbor = neighbors[i];
#pragma omp critical
            {
                if (!visited[neighbor])
                {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
    }
}

void parallelDFS(Graph &graph, int source, vector<bool> &visited, vector<int> &dfs_order)
{
    stack<int> s;
    s.push(source);
    visited[source] = true;

    while (!s.empty())
    {
        int current = s.top();
        s.pop();
        dfs_order.push_back(current);

        vector<int> neighbors = graph.getNeighbors(current);
#pragma omp parallel for
        for (int i = 0; i < neighbors.size(); ++i)
        {
            int neighbor = neighbors[i];
#pragma omp critical
            {
                if (!visited[neighbor])
                {
                    visited[neighbor] = true;
                    s.push(neighbor);
                }
            }
        }
    }
}

int main()
{
    int V, E;
    cout << "Enter the number of vertices: ";
    cin >> V;
    Graph graph(V);

    cout << "Enter the number of edges: ";
    cin >> E;
    cout << "Enter the edges (src dest):" << endl;
    for (int i = 0; i < E; ++i)
    {
        int src, dest;
        cin >> src >> dest;
        graph.addEdge(src, dest);
    }

    vector<bool> visited(V, false);
    vector<int> bfs_order, dfs_order;

    cout << "\nParallel BFS:" << endl;
#pragma omp parallel num_threads(2)
    {
#pragma omp single nowait
        parallelBFS(graph, 0, visited, bfs_order);
    }

    cout << "BFS Order: ";
    for (int node : bfs_order)
        cout << node << " ";
    cout << endl;

    // Reset visited array for DFS
    fill(visited.begin(), visited.end(), false);

    cout << "\nParallel DFS:" << endl;
#pragma omp parallel num_threads(2)
    {
#pragma omp single nowait
        parallelDFS(graph, 0, visited, dfs_order);
    }

    cout << "DFS Order: ";
    for (int node : dfs_order)
        cout << node << " ";
    cout << endl;

    return 0;
}

/*
Output:

Enter the number of vertices: 5
Enter the number of edges: 6
Enter the edges (src dest):
0 1
0 2
1 3
1 4
2 4
3 4
Parallel BFS:
Visited: 0
Visited: 1
Visited: 2
Visited: 3
Visited: 4
Parallel DFS:
Visited: 0
Visited: 2
Visited: 4
Visited: 3
Visited: 1
*/