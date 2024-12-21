import folium
from folium import plugins
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import networkx as nx

class ChineseRailwayTSP:
    def __init__(self, distance_matrix: np.ndarray, 
                 station_names: List[str],
                 coordinates: List[Tuple[float, float]],
                 train_speeds: Dict[Tuple[int, int], float] = None,
                 station_capacities: Dict[int, int] = None):
        self.distance_matrix = distance_matrix
        self.station_names = station_names
        self.coordinates = coordinates
        self.num_stations = len(distance_matrix)
        self.train_speeds = train_speeds or {}
        self.station_capacities = station_capacities or {}
        
        self.visited = []
        self.unvisited = list(range(self.num_stations))
        self.current_station = 0
        self.total_time = 0.0
        
        self.visited.append(self.current_station)
        self.unvisited.remove(self.current_station)

    def calculate_travel_time(self, from_station: int, to_station: int) -> float:
        distance = self.distance_matrix[from_station][to_station]
        speed = self.train_speeds.get((from_station, to_station), 300)
        return distance / speed

    def calculate_station_costs(self) -> List[Tuple[int, float]]:
        costs = []
        for station in self.unvisited:
            travel_time = self.calculate_travel_time(self.current_station, station)
            capacity_penalty = 0
            if station in self.station_capacities:
                capacity_penalty = 0.1 * (1000 / max(1, self.station_capacities[station]))
            distance_from_origin = self.distance_matrix[0][station]
            distance_penalty = 0.001 * distance_from_origin
            total_cost = travel_time + capacity_penalty + distance_penalty
            costs.append((station, total_cost))
        return sorted(costs, key=lambda x: x[1])

    def optimize_route(self) -> Tuple[List[int], float]:
        while self.unvisited:
            costs = self.calculate_station_costs()
            next_station = costs[0][0]
            next_cost = costs[0][1]
            self.total_time += next_cost
            self.current_station = next_station
            self.visited.append(next_station)
            self.unvisited.remove(next_station)
        
        final_time = self.calculate_travel_time(self.current_station, self.visited[0])
        self.total_time += final_time
        self.visited.append(self.visited[0])
        return self.visited, self.total_time

    def visualize_interactive_map(self, route=None, save_html=True):
        """Generate interactive map visualization"""
        m = folium.Map(location=[35.8617, 104.1954], zoom_start=4,
                      tiles='cartodb positron')

        # Add stations
        for i, (lat, lon) in enumerate(self.coordinates):
            color = 'red' if i == 0 else 'blue'
            size = 10 if i == 0 else 8
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=size,
                popup=f"{self.station_names[i]}<br>Capacity: {self.station_capacities.get(i, 'N/A')}",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

        # Draw route with information
        if route:
            for i in range(len(route)-1):
                start_idx = route[i]
                end_idx = route[i+1]
                start_coord = self.coordinates[start_idx]
                end_coord = self.coordinates[end_idx]
                
                # Get route information
                distance = self.distance_matrix[start_idx][end_idx]
                speed = self.train_speeds.get((start_idx, end_idx), 300)
                time = distance / speed
                
                # Create info popup
                info = f"""
                {self.station_names[start_idx]} → {self.station_names[end_idx]}
                Distance: {distance:.1f} km
                Speed: {speed} km/h
                Time: {time:.1f} hours
                """
                
                # Draw line with info
                folium.PolyLine(
                    locations=[start_coord, end_coord],
                    weight=3,
                    color='red',
                    opacity=0.8,
                    popup=info
                ).add_to(m)

                # Add direction arrow
                folium.plugins.AntPath(
                    locations=[start_coord, end_coord],
                    weight=2,
                    color='blue',
                    opacity=0.6
                ).add_to(m)

        if save_html:
            m.save('china_railway_network.html')
        return m

    def visualize_network_analysis(self, route=None):
        """Generate multiple visualization plots"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Network Graph
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        G = nx.Graph()
        for i in range(self.num_stations):
            G.add_node(i, pos=(self.coordinates[i][1], self.coordinates[i][0]))
        
        for i in range(self.num_stations):
            for j in range(i+1, self.num_stations):
                if self.distance_matrix[i][j] > 0:
                    G.add_edge(i, j, weight=self.distance_matrix[i][j])

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_nodes(G, pos, node_color='red', node_size=200)
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.2)
        nx.draw_networkx_labels(G, pos, 
                              {i: name for i, name in enumerate(self.station_names)}, 
                              font_size=8)
        
        if route:
            path_edges = list(zip(route[:-1], route[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                                 edge_color='blue', width=2)
        ax1.set_title("Railway Network Graph")

        # 2. Distance Heatmap
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        sns.heatmap(self.distance_matrix, 
                   xticklabels=self.station_names,
                   yticklabels=self.station_names,
                   cmap='YlOrRd',
                   ax=ax2)
        ax2.set_title("Distance Matrix Heatmap")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # 3. Station Capacities Bar Chart
        ax3 = plt.subplot2grid((2, 2), (1, 0))
        capacities = [self.station_capacities.get(i, 0) for i in range(self.num_stations)]
        sns.barplot(x=self.station_names, y=capacities, ax=ax3)
        ax3.set_title("Station Capacities")
        plt.xticks(rotation=45)

        # 4. Route Statistics
        if route:
            ax4 = plt.subplot2grid((2, 2), (1, 1))
            route_distances = []
            for i in range(len(route)-1):
                route_distances.append(self.distance_matrix[route[i]][route[i+1]])
            
            sns.lineplot(x=range(len(route_distances)), y=route_distances, ax=ax4)
            ax4.set_title("Distance between consecutive stations in route")
            ax4.set_xlabel("Step")
            ax4.set_ylabel("Distance (km)")

        plt.tight_layout()
        plt.savefig('railway_analysis.png', dpi=300, bbox_inches='tight')
        return plt

def main():
    # 30 Chinese cities with their coordinates (latitude, longitude)
    stations = [
        "Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Wuhan",
        "Chengdu", "Xian", "Nanjing", "Hangzhou", "Tianjin",
        "Chongqing", "Zhengzhou", "Jinan", "Qingdao", "Changsha",
        "Nanchang", "Fuzhou", "Hefei", "Kunming", "Harbin",
        "Shenyang", "Dalian", "Xiamen", "Ningbo", "Suzhou",
        "Wenzhou", "Guiyang", "Nanning", "Lanzhou", "Urumqi"
    ]
    
    coordinates = [
        (39.9042, 116.4074),  # Beijing
        (31.2304, 121.4737),  # Shanghai
        (23.1291, 113.2644),  # Guangzhou
        (22.5431, 114.0579),  # Shenzhen
        (30.5928, 114.3055),  # Wuhan
        (30.5728, 104.0668),  # Chengdu
        (34.3416, 108.9398),  # Xian
        (32.0603, 118.7969),  # Nanjing
        (30.2741, 120.1551),  # Hangzhou
        (39.3434, 117.3616),  # Tianjin
        (29.5630, 106.5516),  # Chongqing
        (34.7472, 113.6249),  # Zhengzhou
        (36.6512, 117.1201),  # Jinan
        (36.0671, 120.3826),  # Qingdao
        (28.2278, 112.9388),  # Changsha
        (28.6820, 115.8579),  # Nanchang
        (26.0745, 119.2965),  # Fuzhou
        (31.8206, 117.2272),  # Hefei
        (25.0389, 102.7183),  # Kunming
        (45.8038, 126.5340),  # Harbin
        (41.8057, 123.4315),  # Shenyang
        (38.9140, 121.6147),  # Dalian
        (24.4798, 118.0819),  # Xiamen
        (29.8683, 121.5440),  # Ningbo
        (31.2990, 120.5853),  # Suzhou
        (27.9994, 120.6668),  # Wenzhou
        (26.6470, 106.6302),  # Guiyang
        (22.8170, 108.3665),  # Nanning
        (36.0611, 103.8343),  # Lanzhou
        (43.8256, 87.6168)    # Urumqi
    ]

    # Station capacities (passengers per hour)
    capacities = {
        0: 250000,  # Beijing
        1: 200000,  # Shanghai
        2: 180000,  # Guangzhou
        3: 150000,  # Shenzhen
        4: 120000,  # Wuhan
        5: 100000,  # Chengdu
        6: 100000,  # Xi'an
        7: 80000,   # Nanjing
        8: 80000,   # Hangzhou
        9: 100000,  # Tianjin
        10: 100000, # Chongqing
        11: 90000,  # Zhengzhou
        12: 70000,  # Jinan
        13: 60000,  # Qingdao
        14: 80000,  # Changsha
        15: 60000,  # Nanchang
        16: 50000,  # Fuzhou
        17: 70000,  # Hefei
        18: 50000,  # Kunming
        19: 80000,  # Harbin
        20: 70000,  # Shenyang
        21: 60000,  # Dalian
        22: 50000,  # Xiamen
        23: 60000,  # Ningbo
        24: 70000,  # Suzhou
        25: 50000,  # Wenzhou
        26: 50000,  # Guiyang
        27: 50000,  # Nanning
        28: 50000,  # Lanzhou
        29: 50000,  # Urumqi
    }

    # Train speeds between major cities (km/h)
    speeds = {
        # Eastern Corridor
        (0, 1): 350,  # Beijing-Shanghai
        (1, 7): 350,  # Shanghai-Nanjing
        (1, 8): 350,  # Shanghai-Hangzhou
        (1, 24): 350, # Shanghai-Suzhou
        (1, 23): 350, # Shanghai-Ningbo
        
        # Southern Corridor
        (2, 3): 350,  # Guangzhou-Shenzhen
        (2, 27): 300, # Guangzhou-Nanning
        (2, 22): 350, # Guangzhou-Xiamen
        
        # Central Corridor
        (0, 4): 350,  # Beijing-Wuhan
        (4, 14): 350, # Wuhan-Changsha
        (4, 15): 300, # Wuhan-Nanchang
        
        # Northern Corridor
        (0, 9): 350,  # Beijing-Tianjin
        (0, 20): 350, # Beijing-Shenyang
        (20, 21): 300, # Shenyang-Dalian
        (20, 19): 300, # Shenyang-Harbin
        
        # Western Corridor
        (5, 10): 300, # Chengdu-Chongqing
        (6, 28): 300, # Xi'an-Lanzhou
        (28, 29): 250, # Lanzhou-Urumqi
        
        # Additional Major Routes
        (0, 11): 350, # Beijing-Zhengzhou
        (0, 12): 350, # Beijing-Jinan
        (12, 13): 350, # Jinan-Qingdao
        (17, 1): 350, # Hefei-Shanghai
        (26, 27): 300, # Guiyang-Nanning
        (16, 22): 350, # Fuzhou-Xiamen
    }
    
    # Add reverse routes with same speeds
    speeds.update({(j, i): v for (i, j), v in speeds.items()})

    # Generate distance matrix (30x30)
    distances = np.zeros((30, 30))
    
    # Calculate distances based on coordinates
    for i in range(30):
        for j in range(30):
            if i != j:
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                # Calculate distance using Haversine formula
                R = 6371  # Earth's radius in kilometers
                
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distances[i][j] = R * c

    # Create and run optimizer
    railway_optimizer = ChineseRailwayTSP(
        distance_matrix=distances,
        station_names=stations,
        coordinates=coordinates,
        train_speeds=speeds,
        station_capacities=capacities
    )

    # Get optimal route
    optimal_route, total_time = railway_optimizer.optimize_route()

    # Generate visualizations
    railway_optimizer.visualize_interactive_map(optimal_route)
    railway_optimizer.visualize_network_analysis(optimal_route)

    # Print results
    print("\n=== Chinese Railway Network Optimization Results (30 Cities) ===")
    print("Optimal Route:", " -> ".join([stations[i] for i in optimal_route]))
    print(f"Total Travel Time: {total_time:.2f} hours")

if __name__ == "__main__":
    main()