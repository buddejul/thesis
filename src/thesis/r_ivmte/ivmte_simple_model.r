# install.packages("ivmte")
# install.packages("splines2")
# install.packages("lpSolveAPI")
# install.packages("lsei ")
# install.packages("AER")

library("AER")
library("ivmte")
knitr::kable(head(AE, n = 10))

lm(data = AE, worked ~ morekids)

lm(data = AE, morekids ~ samesex)

ivreg(data = AE, worked ~ morekids | samesex )

# Simple bounds for ATT using moment approach
results <- ivmte(data = AE,
                 target = "att",
                 m0 = ~ u + yob,
                 m1 = ~ u + yob,
                 ivlike = worked ~ morekids + samesex + morekids*samesex,
                 propensity = morekids ~ samesex + yob,
                 noisy = TRUE)

# Parametric restrictions on m0, m1
args <- list(data = AE,
             ivlike =  worked ~ morekids + samesex + morekids*samesex,
             target = "att",
             m0 = ~ u + I(u^2) + yob + u*yob,
             m1 = ~ u + I(u^2) + I(u^3) + yob + u*yob,
             propensity = morekids ~ samesex + yob)
r <- do.call(ivmte, args)
r
