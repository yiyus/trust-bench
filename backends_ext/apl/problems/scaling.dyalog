Scaling←{
    x1 x2←⍵
    r←(x1-3.0)(⍺×x2- ¯2.0)
    j←2 2⍴1 0 0 ⍺
    r j
}
ScalingHessian←{r j←⍺ Scaling ⍵ ⋄ (⍉j)+.×j}
