Linear←{
    x←⍵
    A←3 2⍴1 0 0 1 1 1
    b←1 2 3
    ((A+.×x)-b)A
}
