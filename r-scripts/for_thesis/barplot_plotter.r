library("base")
library("dplyr")
library("ggplot2")
library("reshape2")

plot_barplot <- function(data_csv, x_axis_col, y_axis_col, title_prefix, title, x_label, y_label, out_img_path) {
  
  data_csv = select(data_csv, c(x_axis_col, y_axis_col))
  
  # data_csv$Batch_Size = factor(data_csv$Batch_Size, c("8", "16", "32", "64", "128", "256"))  
  data_csv[[x_axis_col]] = factor(data_csv[[x_axis_col]], c("8", "16", "32", "64", "128", "256"))
  
  # Stacked barplot with multiple groups
  p <- ggplot(data=data_csv, aes_string(x=x_axis_col, y=y_axis_col))
  p <- p + geom_bar(stat="identity", position=position_dodge(), color="black", fill="blue")
  p <- p + theme_bw()
  p <- p + geom_text(aes_string(x = x_axis_col, y = y_axis_col, label = y_axis_col), position = position_dodge(width = .9), vjust = -0.6, size = 3)
  p <- p + labs(title = paste(title_prefix, title), x = x_label, y = y_label)
  
  ggsave(plot = p, filename = out_img_path, width = 14) 
}

plot_barplot_wrapper <- function(in_csv_path, x_axis_col, y_axis_col, title_prefix, title, x_label, y_label, out_img_path) {
  # Code which checks the parameters
  if (missing(x_axis_col)) {
    x_axis_col = "Batch_Size"
  }
  
  if (missing(y_axis_col)) {
    stop("Must specify the column which is used for the y axis,")
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
  plot_barplot(df, x_axis_col, y_axis_col, title_prefix, title, x_label, y_label, out_img_path)
}

preprocess_column_names <- function(df) {
  column_names = colnames(df)
  column_names = gsub("_$", "", gsub("(\\.+)", "_", column_names, perl=TRUE), perl=TRUE)
  colnames(df) = column_names
  return(df)
}

plot_dir_single_barplot <- function(dir_path, x_axis_col, y_axis_col, title, x_label, y_label) {
  csv_files <- list.files(path = dir_path, pattern = "csv")
  for (csv_file in csv_files) {
    out_img_path <- file.path(dir_path, sub(".csv", ".png", csv_file))
    
    title_prefix <- switch (basename(csv_file),
                            "eth_n16_cps_hp.csv" = "(Eth - 16 Nodes) TensorFlow CPS Hard Placement",
                            "eth_n40_cps_hp.csv" = "(Eth - 40 Nodes) TensorFlow CPS Hard Placement",
                            "inf_n24_cps_hp.csv" = "(Inf - 24 Nodes) TensorFlow CPS Hard Placement",
                            "inf_n4_cps_hp.csv" = "(Inf - 4 Nodes) TensorFlow CPS Hard Placement",
                            "eth_n24_cps_hp.csv" = "(Eth - 24 Nodes) TensorFlow CPS Hard Placement",
                            "eth_n4_cps_hp.csv" = "(Eth - 4 Nodes) TensorFlow CPS Hard Placement",
                            "inf_n2_cps_hp.csv" = "(Inf - 8 Nodes) TensorFlow CPS Hard Placement",
                            "inf_n8_cps_hp.csv" = "(Inf - 8 Nodes) TensorFlow CPS Hard Placement",
                            "eth_n2_cps_hp.csv" = "(Eth - 2 Nodes) TensorFlow CPS Hard Placement",
                            "eth_n8_cps_hp.csv" = "(Eth - 8 Nodes) TensorFlow CPS Hard Placement",
                            "inf_n32_cps_hp.csv" = "(Inf - 32 Nodes) TensorFlow CPS Hard Placement",
                            "eth_n32_cps_hp.csv" = "(Eth - 32 Nodes) TensorFlow CPS Hard Placement",
                            "inf_n16_cps_hp.csv" = "(Inf - 16 Nodes) TensorFlow CPS Hard Placement",
                            "inf_n40_cps_hp.csv" = "(Eth - 40 Nodes) TensorFlow CPS Hard Placement"
    )
    
    tryCatch(
      plot_barplot_wrapper(file.path(dir_path, csv_file), x_axis_col, y_axis_col, title_prefix, title, x_label, y_label, out_img_path),
      error=function(cond) {
        message(cond)
      }
    )
  }
}

# x_axis will likely be 'Batch_Size'

# plot_barplot(
#   data_csv = preprocess_column_names(read.csv('/data/files/university/msc_thesis/csv_results/batch_results/n40_cps_hp.csv')), 
#   x_axis_col = 'Batch_Size', 
#   y_axis_col = 'Forward_Images_Second_40', 
#   title_prefix = '(Inf) TensorFlow CPS',
#   title = 'Forward Images per Second',
#   x_label = "Batch Size",
#   y_label = "Images / Second"
# )

plot_dir_single_barplot(
    dir_path = '/data/files/university/msc_thesis/csv_results/batch_results/',
    x_axis_col = 'Batch_Size',
    y_axis_col = 'Forwd_Images_per_Second_img_sec',
    title = 'Forward Images per Second',
    x_label = "Batch Size",
    y_label = "Images / Second"
)