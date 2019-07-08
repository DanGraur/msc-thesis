library("reshape2")
library("ggplot2")
library("dplyr")

plot_barplot_comparative_network <- function(folder_locations, node_counts, prefix, title_prefix, magnitude_multiplier, save_path, save_dir) {
  if (missing(folder_locations)) {
    stop("Must provide the folder locations")
  }
  
  if (missing(node_counts)) {
    stop("Must provide a list of the corresponging folder locations")
  }
  
  if (length(node_counts) != length(folder_locations)) {
    stop("The lengths of node_counts and folder_locations must be equal")
  }
  
  if (missing(prefix)) {
    warning("prefix is missing; I will assume it's network_traffic")
    prefix = "network_traffic"
  }
  
  if (missing(title_prefix)) {
    title_prefix = ''
  } else {
    title_prefix = paste(title_prefix, "-")
  }
  
  if (missing(magnitude_multiplier)) {
    warning("Numbers will be converted from kbps to Mbps")
    magnitude_multiplier = 1.0 / 1024.0 
  }
  
  accumulated_traffic <- c();
  
  for (folder in folder_locations) {
    csv_files <- list.files(path = folder, pattern = paste(prefix, ".*\\.csv$", sep = ""))
    
    running_total_tx <- 0
    running_total_rx <- 0
    
    for (file in csv_files) {
      current <- read.csv(file.path(folder, file));
      running_total_tx <- running_total_tx + sum(current$tx_kbps);
      running_total_rx <- running_total_rx + sum(current$rx_kbps);
    }
    
    # Here is where we'll convert the kbits to something else
    running_total_rx <- magnitude_multiplier * running_total_rx
    running_total_tx <- magnitude_multiplier * running_total_tx
    
    # We'll add the outgoing and incoming data
    accumulated_traffic <- c(accumulated_traffic, running_total_tx, running_total_rx)
  }
  
  node_counts_stringified <- c()
  
  for (count in node_counts) {
    node_counts_stringified <- c(node_counts_stringified, paste(count, "Nodes"))
  }
  
  final_df <- data.frame("Node Count" = rep(node_counts_stringified, each = 2), "Type" = rep(c("Tx", "Rx"), length(folder_locations)), "Traffic" = accumulated_traffic)
  final_df$Node.Count <- factor(final_df$Node.Count, node_counts_stringified)
  final_df$TrafficGbps <- paste(format(round(final_df$Traffic / 1024.0, 2), nsmall = 2), "Gb") 
  show(final_df)

  p <- ggplot(data=final_df, aes(x=Node.Count, y=Traffic, fill=Type)) + geom_bar(stat="identity", position=position_dodge()) +
    geom_text(aes(x = Node.Count, label = TrafficGbps, y = Traffic), position = position_dodge(width = .9), vjust = -0.6, size = 3)
  p <- p + labs(title = paste(title_prefix, "Incoming / Outgoing Data Comparison"), x = "Node Count", y = "Total Traffic [Mbps]")
  show(p)
  
  if (!missing(save_path) && !missing(save_dir)) {
    ggsave(plot = p, filename = file.path(save_path, save_dir, 'network_traffic_per_nodes_config.png'), width = 7);
  }
}

generate_folder_locations <- function(connection_type, architecture_type) {
  paths <- switch(
    architecture_type,
    "caffe2" = c(
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/caffe2/network/%s/n%d', connection_type, 4),
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/caffe2/network/%s/n%d', connection_type, 8),
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/caffe2/network/%s/n%d', connection_type, 12)
    ),
    "ps_nplus1" = c(
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/%s/n%d/%s', connection_type, 4, "ps_n5"),
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/network/%s/n%d/%s', connection_type, 8, "ps_n9"),
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/%s/n%d/%s', connection_type, 12, "ps_n13"),
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/network/%s/n%d/%s', connection_type, 16, "ps_n17")
    ),
    c(
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/%s/n%d/%s', connection_type, 4, architecture_type),
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/network/%s/n%d/%s', connection_type, 8, architecture_type),
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/cpu_mem_and_network/%s/n%d/%s', connection_type, 12, architecture_type),
      sprintf('/data/files/university/msc_thesis/2nd_semester/r_images/network/%s/n%d/%s', connection_type, 16, architecture_type)
    )
  )
  return(paths)
}

connection_type = "ethernet"
architecture_type = "ps_nplus1"

# for (connection_type in c("ethernet", "infiniband")) {
#   for (architecture_type in c("car", "cps", "dr", "dar_ring", "dar_pscpu", "ps")) {
    # Plot configuration
    save_path = paste('/data/files/university/msc_thesis/2nd_semester/r_images/network_comparison/', connection_type, sep = "")
                      
    title_prefix = switch (
      architecture_type,
      "cps" = "Collocated Parameter Server",
      "ps" = "Parameter Server",
      "ps_n9" = "Parameter Server (9 Nodes)",
      "ps_n17" = "Parameter Server (17 Nodes)",
      "ps_n4" = "Parameter Server (4 Nodes)",
      "ps_n16" = "Parameter Server (16 Nodes)",
      "ps_n5" = "Parameter Server (5 Nodes)",
      "ps_n13" = "Parameter Server (13 Nodes)",
      "dr" = "Distributed Replicated",
      "car" = "Collective All Reduce",
      "dar_pscpu" = "Distributed All Reduce (PSCPU)",
      "dar_ring" = "Distributed All Reduce (RING)",
      "caffe2" = "Caffe2",
      "ps_nplus1" = "Plus 1 Nodes"
    )
     
    node_counts <- switch(
      architecture_type,
      "caffe2" = c(4, 8, 12),
      "ps_nplus1" = c(5, 9, 13, 17),
      c(4, 8, 12, 16)
    )
    
    folder_locations <- generate_folder_locations(connection_type = connection_type, architecture_type = architecture_type)
    
    plot_barplot_comparative_network(folder_locations = folder_locations, node_counts = node_counts, title_prefix = title_prefix, save_path = save_path, save_dir = architecture_type)
#   }
# }