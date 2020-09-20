# library
library(ggplot2)
library(dplyr)

# Create data (this takes more sense with a numerical X axis)
data <- read.csv("../data/features.csv")

# plot
ggplot(data, aes(x=x, y=y)) +
  geom_segment( aes(x=x, xend=x, y=0, yend=y, color=ranking), size=1.3, alpha=0.9) +
  theme_light() +
  theme(
    legend.position = "none",
    panel.border = element_blank(),
  ) +
  xlab("") +
  ylab("Relevance")
