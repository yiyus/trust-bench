IllConditioned←{
    theta←○÷6
    c←2○theta
    s←1○theta
    a←2 2⍴(⍺×c),(-⍺×s),s,c
    xtrue←2 3
    b←a+.×xtrue
    res←(a+.×⍵)-b
    res a
}
IllConditionedHessian←{r j←⍺ IllConditioned ⍵ ⋄ (⍉j)+.×j}
