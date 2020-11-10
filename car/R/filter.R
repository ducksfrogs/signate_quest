library(nycflights13)
filter(flights, month==1)
filter(flights, month==12, day==1)
library(tidyverse)
nov_dec <- filter(flights, month %in% c(11,12))
nov_dec
filter(flights, !(arr_delay >120 | dep_delay >120))
filter(flights, arr_delay <= 120, dep_delay <=120)

df <- tibble(x=c(1,NA,3))
filter(df, x>1)
