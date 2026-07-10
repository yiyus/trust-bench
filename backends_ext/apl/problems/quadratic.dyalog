Quadratic←{x1 x2←⍵ ⋄ (x1(2×x2))(2 2⍴1 0 0 2)}
QuadraticHessian←{r j←Quadratic ⍵ ⋄ (⍉j)+.×j}
