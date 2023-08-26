library('igraph')
library('Matrix')
library('pracma')

#(A)
g <- sample_gnp(n=900, p=0.015, directed=F)

plot(g, vertex.size=1, vertex.label=NA)

#(B)
start <- 1
steps <- 20
iterations <- 1000

avg_distances <- numeric(steps)
variances <- numeric(steps)
end_node_deg <- numeric(steps)

for (t in 1:steps) {
  distances <- numeric(iterations)
  
  for (i in 1:iterations) {
    walk <- random_walk(g, start, t)
    end <- walk[length(walk)]  
    distance <- shortest_paths(g, start, end)$vpath
    distances[i] <- sapply(distance, function(v) length(v))
    #for (C)
    end_node_deg[i] = degree(g, end)
  }
  
  avg_distance <- mean(distances)
  variance <- var(distances)

  avg_distances[t] <- avg_distance
  variances[t] <- variance
}

t <- 1:steps
plot(t, avg_distances)
plot(t, variances)

#(C)
end_node_deg <- unlist(end_node_deg)
hist(end_node_deg, main = "Degree Distribution", xlab = "Degree", ylab = "Frequency")

cat("PoOP")