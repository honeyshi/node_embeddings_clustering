from itertools import combinations, permutations

# Изменение значений кластеров на 1, кроме кластера-исключения
def update_rest_values_by_one(values_dict, except_key, add=False):
  increment = 1 if add else -1
  for key, value in values_dict.items():
    if key != except_key:
      if value == 0 and increment > 0 or value > 0:
        values_dict[key] = value + increment

# Поиск кластера с наибольшим числом голосов. Если таких кластеров несколько, вернуть None и список этих кластеров
def find_cluster_assign(values_dict):
  if len(values_dict) == 1:
    return max(values_dict, key=values_dict.get), []
  max_value = max(values_dict.values())
  keys_with_max = [key for key, value in values_dict.items() if value == max_value]
  if len(keys_with_max) > 1:
    return None, keys_with_max
  else:
    if max_value == 0:
      return None, keys_with_max
    return keys_with_max[0], []

# Получение кластера для не рассмотренного узла из рассмотренного. Если кластеров с максимум несколько, вернуть их
def get_existing_node_cluster(values_dict):
  node_cluster, keys_with_max = find_cluster_assign(values_dict)
  if node_cluster == None:
    return {key: values_dict[key] for key in keys_with_max}
  else:
    return {node_cluster: 1}
  
# После посещения ранее не рассмторенного узла, необходимо увеличить вес его кластера
def update_weight_after_adding_unseen_node(values_dict, node_cluster_dict):
  if len(node_cluster_dict) == 1:
    node_cluster = next(iter(node_cluster_dict))
    values_dict[node_cluster] += 1

# Добавить в кластеры назначение для расмотренной и не рассмотренной вершин
def add_nodes_pair_seen_unseen(cluster_assign, first_node, second_node):
  second_node_cluster = get_existing_node_cluster(cluster_assign[first_node])
  cluster_assign[second_node] = second_node_cluster
  update_weight_after_adding_unseen_node(cluster_assign[first_node], second_node_cluster)

# Добавить в кластеры назначение для расмотренной и не рассмотренной вершин не из одного кластера
def add_nodes_pair_seen_unseen_not_in(cluster_assign, first_node, second_node, current_cluster):
  node_cluster, _ = find_cluster_assign(cluster_assign[first_node])
  if node_cluster != None:
    cluster_assign[first_node][node_cluster] += 1
  cluster_assign[second_node] = {current_cluster: 1}

# Подтвердить назначение кластеру, увеличив его на 1, а остальные назначения уменьшив на 1
def accept_assign_to_cluster(cluster_assign, node, node_cluster):
  cluster_assign[node][node_cluster] += 1
  update_rest_values_by_one(cluster_assign[node], node_cluster)

# Опровергнуть назначение кластеру, уменьшив его на 1, а остальные назначения увеличив на 1
def discard_assign_to_cluster(cluster_assign, node, node_cluster):
  if cluster_assign[node][node_cluster] > 0:
    cluster_assign[node][node_cluster] -= 1
  update_rest_values_by_one(cluster_assign[node], node_cluster, add=True)

# Добавить новый кластер или увеличить его значение, а остальные назначения уменьшить на 1
# Если передана еще одна вершина, для нее значение этого кластера уменьшить на 1
def add_new_cluster(cluster_assign, node, new_cluster, another_node=None):
  if another_node != None and cluster_assign[another_node][new_cluster] > 0:
    cluster_assign[another_node][new_cluster] -= 1
  if new_cluster in cluster_assign[node]:
    cluster_assign[node][new_cluster] += 1
  else:
    cluster_assign[node][new_cluster] = 1
  update_rest_values_by_one(cluster_assign[node], new_cluster)

# Уменьшает значение кластера для одной вершины, а другой увеличивает
def downvote_cluster(cluster_assign, node, new_cluster, another_node=None):
  cluster_assign[another_node][new_cluster] += 1
  if new_cluster in cluster_assign[node] and cluster_assign[node][new_cluster] > 0:
    cluster_assign[node][new_cluster] -= 1
  update_rest_values_by_one(cluster_assign[node], new_cluster, add=True)

# Увеличивает значение кластера той вершине, где он максимален.
# Для другой убираем остальные назначения и оставляем только этот кластер.
def assign_to_max_cluster(cluster_assign, first_node, second_node, max_cluster):
  cluster_assign[first_node][max_cluster] += 1
  if max_cluster in cluster_assign[second_node]:
    cluster_assign[second_node][max_cluster] += 1
  else:
    cluster_assign[second_node][max_cluster] = 1
  cluster_value = cluster_assign[second_node][max_cluster]
  cluster_assign[second_node].clear()
  cluster_assign[second_node][max_cluster] = cluster_value

def perform_clustering(G, pairs_classification, with_permutations):
  cluster_assign = {}
  current_cluster = 0

  for index, pair in enumerate(permutations(G.nodes(), 2) if with_permutations else combinations(G.nodes(), 2)):
    first_node = pair[0]
    second_node = pair[1]

    # Обе вершины классифицированы как принадлежащие одному кластеру
    if pairs_classification[index] == 1:
      # Обе вершины уже были встречены ранее
      if first_node in cluster_assign and second_node in cluster_assign:
        # Получим их текущие назначения кластеров
        first_node_cluster, _ = find_cluster_assign(cluster_assign[first_node])
        second_node_cluster, _ = find_cluster_assign(cluster_assign[second_node])
        # Если назначения кластеров равны и они оба не None, то мы подтверждаем назначение этому кластеру,
        # то есть увеличиваем его на 1, а остальные назначения уменьшаем на 1
        if first_node_cluster == second_node_cluster and first_node_cluster != None:
          accept_assign_to_cluster(cluster_assign, first_node, first_node_cluster)
          accept_assign_to_cluster(cluster_assign, second_node, second_node_cluster)
        # Назначения кластерам не равны
        if first_node_cluster != second_node_cluster:
          # Оба кластера известны, тогда assign_to_max_cluster: Отнести вершину к кластеру с максимумом
          if first_node_cluster != None and second_node_cluster != None:
            if cluster_assign[first_node][first_node_cluster] > cluster_assign[second_node][second_node_cluster]:
              assign_to_max_cluster(cluster_assign, first_node, second_node, first_node_cluster)
            elif cluster_assign[first_node][first_node_cluster] < cluster_assign[second_node][second_node_cluster]:
              assign_to_max_cluster(cluster_assign, second_node, first_node, second_node_cluster)
          # Один кластер известен, а другой нет. Тогда для вершины с известным кластером вес уменьшаем, 
          # так как эта ситуация не дает нам уверенности в принадлежности к этому кластеру. 
          # Для вершины с неизвестным кластером увеличиваем (или добавляем) вес известному кластеру, остальные уменьшаем
          if first_node_cluster != None and second_node_cluster == None:
            add_new_cluster(cluster_assign, second_node, first_node_cluster, first_node)
          if first_node_cluster == None and second_node_cluster != None:
            add_new_cluster(cluster_assign, first_node, second_node_cluster, second_node)
      # Обеим вершинам не были назначены кластеры
      if first_node not in cluster_assign and second_node not in cluster_assign:
        cluster_assign[first_node] = {current_cluster: 1}
        cluster_assign[second_node] = {current_cluster: 1}
        current_cluster += 1
      # Первой вершине был назначен кластер, а второй еще нет
      if first_node in cluster_assign and second_node not in cluster_assign:
        add_nodes_pair_seen_unseen(cluster_assign, first_node, second_node)
      # Второй вершине был назначен кластер, а первой еще нет
      if second_node in cluster_assign and first_node not in cluster_assign:
        add_nodes_pair_seen_unseen(cluster_assign, second_node, first_node)

    # Вершины не были классифицированы как принадлежащие одному кластеру
    if pairs_classification[index] == 0:
      # Обе вершины уже были встречены ранее
      if first_node in cluster_assign and second_node in cluster_assign:
        # Получим их текущие назначения кластеров
        first_node_cluster, _ = find_cluster_assign(cluster_assign[first_node])
        second_node_cluster, _ = find_cluster_assign(cluster_assign[second_node])
        # Если назначения кластеров равны и они оба не None, то мы уменьшаем назначение этому кластеру, 
        # а остальные назначения увеличиваем на 1. При этом второй вершине назначаем следующий
        # после наибольшего для первой вершины кластер
        if first_node_cluster == second_node_cluster and first_node_cluster != None:
          discard_assign_to_cluster(cluster_assign, first_node, first_node_cluster)
          discard_assign_to_cluster(cluster_assign, second_node, second_node_cluster)
          if first_node_cluster + 1 not in cluster_assign[second_node]:
            cluster_assign[second_node][first_node_cluster + 1] = 1
          else:
            cluster_assign[second_node][first_node_cluster + 1] += 1
        # Назначения кластерам не равны
        if first_node_cluster != second_node_cluster:
          # Один кластер известен, а другой нет. Тогда для вершины с известным кластером вес увеличиваем.
          # Для вершины с неизвестным кластером уменьшаем вес известному кластеру, остальные увеличиваем
          if first_node_cluster != None and second_node_cluster == None:
            downvote_cluster(cluster_assign, second_node, first_node_cluster, first_node)
          if first_node_cluster == None and second_node_cluster != None:
            downvote_cluster(cluster_assign, first_node, second_node_cluster, second_node)
      # Обеим вершинам не были назначены кластеры
      if first_node not in cluster_assign and second_node not in cluster_assign:
        cluster_assign[first_node] = {current_cluster: 1}
        cluster_assign[second_node] = {current_cluster + 1: 1}
        current_cluster += 2
      # Первой вершине был назначен кластер, а второй еще нет, 
      # назначить вершине следующий кластер после макимального для первой
      if first_node in cluster_assign and second_node not in cluster_assign:
        first_node_cluster, _ = find_cluster_assign(cluster_assign[first_node])
        if first_node_cluster != None:
          cluster_assign[second_node] = {first_node_cluster + 1: 1}
        else:
          cluster_assign[second_node] = {current_cluster: 1}
          current_cluster += 1
      # Второй вершине был назначен кластер, а первой еще нет
      if second_node in cluster_assign and first_node not in cluster_assign:
        second_node_cluster, _ = find_cluster_assign(cluster_assign[second_node])
        if second_node_cluster != None:
          cluster_assign[first_node] = {second_node_cluster + 1: 1}
        else:
          cluster_assign[first_node] = {current_cluster: 1}
          current_cluster += 1

  return cluster_assign

def get_nodes_clusters(cluster_assign):
  nodes_clusters = {}
  for node, assign in cluster_assign.items():
    nodes_clusters[node] = max(assign, key=assign.get)

  return nodes_clusters

def restore_missing_clusters(cluster_assign):
  clusters = list(cluster_assign.values())
  full_sequence = set(range(0, clusters[-1] + 1))
  clusters_set = set(clusters)
  missing_clusters = list(clusters_set ^ full_sequence)

  for key, value in cluster_assign.items():
    for missing in missing_clusters:
      if value > missing:
        cluster_assign[key] -= 1
      else:
        break

def map_nodes_to_clusters(nodes_clusters):
  clusters = {}
  for key, value in nodes_clusters.items():
    if value in clusters:
      clusters[value].append(key)
    else:
      clusters[value] = [key]

  return clusters

def clustering_pipeline(G, pairs_classification, with_permutations):
  cluster_assign = perform_clustering(G, pairs_classification, with_permutations)
  nodes_clusters = get_nodes_clusters(cluster_assign)
  restore_missing_clusters(nodes_clusters)
  clusters = map_nodes_to_clusters(nodes_clusters)
  return clusters