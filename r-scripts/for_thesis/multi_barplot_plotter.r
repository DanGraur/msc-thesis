library("base")
library("dplyr")
library("ggplot2")
library("reshape2")

plot_vs_barplot_different_file <- function(timeseries_dfs, x_axis_col, y_axis_col, title_prefix, title, x_label, y_label, nice_column_names, out_img_path) {
  # This method plots barplots in groups containing an element from each df on the y_axis_col on the x_axis_col positions.
  data_column = c()
  
  # Get the columns that have been selected for plotting, and merge them into one column 
  for (df in timeseries_dfs) {
    data_column = c(data_column, df[[y_axis_col]])  
  }
  
  x_axis_elements = timeseries_dfs[[1]][[x_axis_col]]
  
  final_df <- data.frame(x_axis_col = rep(x_axis_elements, length(timeseries_dfs)), "Type" = rep(nice_column_names, each=length(timeseries_dfs)), "Values" = data_column)
  final_df[[x_axis_col]] = factor(x_axis_elements, c("8", "16", "32", "64", "128", "256"))
  
  print(final_df)
  
  # Stacked barplot with multiple groups
  p <- ggplot(data=final_df, aes_string(x=x_axis_col, y="Values", fill="Type"))
  p <- p + geom_bar(stat="identity", position=position_dodge())
  p <- p + theme_bw()
  # Uncomenting the line below might lead to overlapping text
  # p <- p + geom_text(aes_string(x = x_axis_col, y="Values", label="Values"), position = position_dodge(width = .9), vjust = -0.6, size = 3)
  p <- p + labs(title = paste(title_prefix, title), x = x_label, y = y_label)
  
  show(p)
  ggsave(plot = p, filename = out_img_path, width = 14)  
  
}

plot_vs_barplot_same_file <- function(timeseries_df, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path) {
  # This method plots barplots in groups containing an element from each y_axis_cols element on the x_axis_col positions. The elements belong to the same file     

  timeseries_df = select(timeseries_df, c(x_axis_col, y_axis_cols))
  data_column = c()
  
  # Get the columns that have been selected for plotting, and merge them into one column 
  for (col_name in y_axis_cols) {
    data_column = c(data_column, timeseries_df[[col_name]])  
  }
  
  final_df <- data.frame(x_axis_col = rep(timeseries_df[[x_axis_col]], length(y_axis_cols)), "Type" = rep(nice_column_names, each = length(timeseries_df[[x_axis_col]])), "Values" = data_column)
  final_df[[x_axis_col]] = factor(timeseries_df[[x_axis_col]], c("8", "16", "32", "64", "128", "256"))
  
  print(final_df)
  
  # Stacked barplot with multiple groups
  p <- ggplot(data=final_df, aes_string(x=x_axis_col, y="Values", fill="Type"))
  p <- p + geom_bar(stat="identity", position=position_dodge())
  p <- p + theme_bw()
  # Uncomenting the line below might lead to overlapping text
  # p <- p + geom_text(aes_string(x = x_axis_col, y="Values", label="Values"), position = position_dodge(width = .9), vjust = -0.6, size = 3)
  p <- p + labs(title = paste(title_prefix, title), x = x_label, y = y_label)

  show(p)
  ggsave(plot = p, filename = out_img_path, width = 14)
}

plot_vs_barplot_same_file_wrapper <- function(in_csv_path, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path) {
  # Code which checks the parameters
  if (missing(x_axis_col)) {
    x_axis_col = "Batch_Size"
  }
  
  if (missing(y_axis_cols)) {
    stop("Must specify the columns which are used for the y axis,")
  }
  
  # If not prefix is specified, default to ''. If it is specified, add a ' - ' after it.
  if (missing(title_prefix)) {
    title_prefix = ''
  } else {
    title_prefix = paste(title_prefix, " - ")
  }
  
  if (missing(title)) {
    stop("Title must be specified")
  }
  
  if (missing(x_label)) {
    x_label = "Batch Size"
    warning("No X label specified. Will default to: 'Node Count'")
  }
  
  if (missing(y_label)) {
    stop("Must specify a Y label")
  }
  
  if (missing(out_img_path)) {
    stop("Need to specify the path to the stored image")
  }
  
  # Actual code
  df <- read.csv(in_csv_path)
  df <- preprocess_column_names(df)
  plot_vs_barplot_same_file(df, x_axis_col, y_axis_cols, title_prefix, title, x_label, y_label, nice_column_names, out_img_path)
}

preprocess_column_names <- function(df) {
  column_names = colnames(df)
  column_names = gsub("_$", "", gsub("(\\.+)", "_", column_names, perl=TRUE), perl=TRUE)
  colnames(df) = column_names
  return(df)
}

# plot_vs_barplot_same_file_wrapper(
#   in_csv_path = '/data/files/university/msc_thesis/csv_results/batch_results/eth_n2_cps_hp.csv',
#   x_axis_col = 'Batch_Size',
#   y_axis_cols = c('Forwd_Images_per_Second_img_sec', 'Forwd_Backwd_Images_per_Second_img_sec', 'Forwd_Backwd_Seconds_per_Batch_sec_batch'),
#   title_prefix = "Asd",
#   title = 'Forward Images per Second vs.Forward and Bacward Images per Second',
#   x_label = "Batch Size",
#   y_label = "Images / Second",
#   nice_column_names = c("Forward Images per Second", "Forward and Backward Images per Second", "Forward and Backward Seconds per Batch"),
#   out_img_path = '/data/files/university/msc_thesis/csv_results/batch_results/eth_n2_cps_hp_test.png'
# )

a = preprocess_column_names(read.csv('/data/files/university/msc_thesis/csv_results/batch_results/eth_n4_cps_hp.csv'))
b = preprocess_column_names(read.csv('/data/files/university/msc_thesis/csv_results/batch_results/eth_n8_cps_hp.csv'))
c = preprocess_column_names(read.csv('/data/files/university/msc_thesis/csv_results/batch_results/eth_n16_cps_hp.csv'))
d = list(a, b, c)

plot_vs_barplot_different_file(
    timeseries_dfs = d,
    x_axis_col = 'Batch_Size',
    y_axis_col = 'Forwd_Images_per_Second_img_sec',
    title_prefix = "Asd",
    title = 'Forward Images per Second vs.Forward and Bacward Images per Second',
    x_label = "Batch Size",
    y_label = "Forward Images / Second",
    nice_column_names = c("4 Nodes", "8 Nodes", "16 Nodes"),
    out_img_path = '/data/files/university/msc_thesis/csv_results/batch_results/one_col_vs.png'
)
