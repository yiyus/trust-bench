Outliers←{
    n←20
    t←(¯1+⍳n)÷n-1
    ncorrupt←⌊0.5+⍺×n
    y←(2.0×t)+1.0
    corrmask←(⍳n)≤ncorrupt
    y←y+corrmask×5.0
    m c←⍵
    r←((m×t)+c)-y
    j←⍉2 n⍴t,n⍴1.0
    r j
}
OutliersHessian←{r j←⍺ Outliers ⍵ ⋄ (⍉j)+.×j}
