_MichaelisMentenS←0.5 1.0 2.0 4.0 6.0 8.0 10.0 15.0
_MichaelisMentenY←1.4933620489887474 2.546932079438029 3.935697938925916 5.596459849435202 6.6521976259771405 7.393073112346602 7.825666073598016 8.424163473662125
MichaelisMenten←{
    vmax km←⍵
    denom←km+_MichaelisMentenS
    r←(vmax×_MichaelisMentenS÷denom)-_MichaelisMentenY
    j←⍉2 8⍴(_MichaelisMentenS÷denom),-vmax×_MichaelisMentenS÷denom*2
    r j
}
MichaelisMentenHessian←{
    vmax km←⍵
    r j←MichaelisMenten ⍵
    denom←km+_MichaelisMentenS
    h←(⍉j)+.×j
    d2vdkm2←2×vmax×_MichaelisMentenS÷denom*3
    d2vdvmaxdkm←-_MichaelisMentenS÷denom*2
    s←+/r×d2vdvmaxdkm
    h[1;2]+←s ⋄ h[2;1]+←s
    h[2;2]+←+/r×d2vdkm2
    h
}
