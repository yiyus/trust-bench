Helical←{
    a b c←⍵
    T←{1|1+(12○⍺+0J1×⍵)÷○2} ⋄ M←{|⍺+0J1×⍵}
    m←a M b ⋄ y←(10×c-10×a T b)(10×m-1)c
    y[((50×b)÷○+.×⍨a b)(-(50×a)÷○+.×⍨a b)10 ⋄ (10×a÷m)(10×b÷m)0 ⋄ 0 0 1]
}
HelicalHessian←{
    a b c←⍵
    r j←Helical ⍵
    h←(⍉j)+.×j
    rr←(a×a)+b×b
    pi←○1
    d2t11←(2×a×b÷rr×rr)÷2×pi
    d2t22←(-2×a×b÷rr×rr)÷2×pi
    d2t12←((b×b)-a×a)÷(rr×rr)×2×pi
    h1←3 3⍴(-100×d2t11)(-100×d2t12)0(-100×d2t12)(-100×d2t22)0 0 0 0
    s←rr*0.5
    d2s11←(b×b)÷s*3
    d2s22←(a×a)÷s*3
    d2s12←(-a×b)÷s*3
    h2←3 3⍴(10×d2s11)(10×d2s12)0(10×d2s12)(10×d2s22)0 0 0 0
    h+(r[1]×h1)+(r[2]×h2)
}
