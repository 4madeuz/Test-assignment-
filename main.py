import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class CityGrid:

    def __init__(self, rows, cols, obstruction_prob=0.3, tower_cost=1, budget=10):
        self.rows = rows
        self.cols = cols
        self.grid = np.random.choice([0, 1], size=(rows, cols), p=[1-obstruction_prob, obstruction_prob])
        self.tower_cost = tower_cost
        self.budget = budget

    def place_tower(self, row, col, range_R):
        '''Отмечает центр башни цифрой 3 и покрытие башни цифрами 2'''
        for i in range(max(0, row - range_R), min(self.rows, row + range_R + 1)):
            for j in range(max(0, col - range_R), min(self.cols, col + range_R + 1)):
                self.grid[i, j] = 2
        self.grid[row, col] = 3

    def calculate_total_cost(self):
        return np.count_nonzero(self.grid == 3) * self.tower_cost

    def place_optimal_towers(self, range_R):
        '''Пока позволяет бюджет ищем блок с минимальным покрытием и ставим башню'''
        while self.calculate_total_cost() + self.tower_cost <= self.budget:
            max_uncovered_block = self.find_max_uncovered_block(range_R)
            if max_uncovered_block is None:
                break  # No more blocks to cover
            row, col = max_uncovered_block
            self.place_tower(row, col, range_R)

        print("Optimal towers placed.")

    def find_max_uncovered_block(self, range_R):
        '''Ищем допустимое место для башни и считаем количество блоков без покрытия, возвращаем координаты для постройки башни'''
        uncovered_blocks = np.argwhere(self.grid == 0)
        max_uncovered_count = 0
        max_uncovered_block = None

        for block in uncovered_blocks:
            row, col = block
            count = self.count_uncovered_neighbors(row, col, range_R)
            if count > max_uncovered_count:
                max_uncovered_count = count
                max_uncovered_block = (row, col)

        return max_uncovered_block

    def count_uncovered_neighbors(self, row, col, range_R):
        '''Подсчёт количества блоков без покрытия'''
        count = 0
        for i in range(max(0, row - range_R), min(self.rows, row + range_R + 1)):
            for j in range(max(0, col - range_R), min(self.cols, col + range_R + 1)):
                if self.grid[i, j] == 0:
                    count += 1
        return count

    def count_neighbor_towers(self, row, col, range_R):
        '''Подсчет количества башен в радиусе'''
        towers = []
        for i in range(max(0, row - range_R), min(self.rows, row + range_R + 1)):
            for j in range(max(0, col - range_R), min(self.cols, col + range_R + 1)):
                if self.grid[i, j] == 3:
                    towers.append((i, j))
        return towers

    def build_tower_graph(self, range_R):
        '''Создаём граф с вершинами в башнях = 3, если радиусы башен пересекаются, соеденяем вершины ребрами'''
        tower_graph = nx.Graph()

        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 3:
                    tower_graph.add_node((i, j))
                    towers = self.count_neighbor_towers(i, j, range_R*2)
                    for tower in towers:
                        if tower != (i, j):
                            tower_graph.add_edge((i, j), tower)

        return tower_graph

    def find_reliable_path(self, start, end, graph):
        '''Поиск оптимального пути'''
        try:
            path = nx.shortest_path(graph, source=start, target=end)
            return path
        except nx.NetworkXNoPath:
            return None

    def towers(self):
        '''Получаем первую и последнюю башню'''
        existing_towers = np.argwhere(self.grid == 3)
        return existing_towers[0], existing_towers[-1]

    def visualize_graph(self, graph):
        pos = {node: (node[1], -node[0]) for node in graph.nodes()}
        nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_color='black', font_weight='bold')
        plt.title("Tower Graph")
        plt.show()

    def visualize_path(self, path, graph):
        pos = {node: (node[1], -node[0]) for node in graph.nodes()}
        nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_color='black', font_weight='bold')
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_size=700, node_color='red')
        nx.draw_networkx_edges(graph, pos, edgelist=list(zip(path, path[1:])), edge_color='red', width=2)
        plt.title("Most Reliable Path")
        plt.show()

    def visualize_optimal_towers(self):
        tower_centers = np.argwhere(self.grid == 3)
        tower_centers_x, tower_centers_y = tower_centers[:, 1], tower_centers[:, 0]

        plt.scatter(tower_centers_x, tower_centers_y, color='red', marker='x', label='Tower Centers')
        plt.imshow(self.grid, cmap='viridis', interpolation='nearest', alpha=0.3)
        plt.title("Optimal Tower Placement")
        plt.legend()

        total_cost = self.calculate_total_cost()
        print(f"Total cost of towers: {total_cost}")

        plt.show()

    def visualize_grid(self):
        print(self.grid)


city = CityGrid(20, 20, tower_cost=2, budget=200)
city.visualize_grid()
city.place_optimal_towers(2)
city.visualize_optimal_towers()
city.visualize_grid()
tower1, tower2 = city.find_random_existing_towers()
print("First Tower 1:", tower1)
print("Last Tower 2:", tower2)
graph = city.build_tower_graph(2)
city.visualize_graph(graph)
reliable_path = city.find_reliable_path(tuple(tower1), tuple(tower2), graph)
if reliable_path:
    city.visualize_path(reliable_path, graph)
