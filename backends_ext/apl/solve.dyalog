∇res←Solve req;f;hf;method;calls;hcalls;fd;lower;cfg;r;gnorm;bounded;status;parsed;family;param;isParam;t0;solveMs
  f←NameFor req.problem_id
  isParam←0
  :If 0=≢f
      parsed←ParseParametrised req.problem_id
      :If 0≠≢parsed
          family param←parsed
          f←FamilyNameFor family
          isParam←1
      :EndIf
      :If 0=≢f
          res←ErrorResult'Unknown problem_id: ',req.problem_id
          :Return
      :EndIf
  :EndIf
  method←{2=⎕NC'req.method':req.method ⋄ 'lm'}⍬
  :If ~(method≡'lm')∨(method≡'BFGS')∨(method≡'trust-exact')
      res←ErrorResult'Unknown method: ',method
      :Return
  :EndIf
  calls←0
  hcalls←0
  fd←0
  :If 2=⎕NC'req.derivative_mode'
      fd←req.derivative_mode≡'finite-difference'
  :EndIf
  :If method≡'lm'
      :If isParam
          lower←{calls+←1 ⋄ result←ApplyParam f param ⍵ ⋄ fd:1⊃result ⋄ result}
      :Else
          lower←{calls+←1 ⋄ result←Apply f ⍵ ⋄ fd:1⊃result ⋄ result}
      :EndIf
  :ElseIf method≡'BFGS'
      :If isParam
          lower←{
              calls+←1
              r j←ApplyParam f param ⍵
              fd:0.5×+/r×r
              (0.5×+/r×r)((⍉j)+.×r)
          }
      :Else
          lower←{
              calls+←1
              r j←Apply f ⍵
              fd:0.5×+/r×r
              (0.5×+/r×r)((⍉j)+.×r)
          }
      :EndIf
  :Else
      :If isParam
          hf←FamilyHessianNameFor family
          lower←{
              calls+←1
              r j←ApplyParam f param ⍵
              hcalls+←1
              hess←ApplyParam hf param ⍵
              (0.5×+/r×r)hess((⍉j)+.×r)
          }
      :Else
          hf←HessianNameFor req.problem_id
          lower←{
              calls+←1
              r j←Apply f ⍵
              hcalls+←1
              hess←Apply hf ⍵
              (0.5×+/r×r)hess((⍉j)+.×r)
          }
      :EndIf
  :EndIf
  cfg←⎕NS''
  cfg.loss←{2=⎕NC'req.loss':req.loss ⋄ 'L2'}⍬
  cfg.toli←{2=⎕NC'req.max_iter':req.max_iter ⋄ 1E3}⍬
  cfg.tolc←{2=⎕NC'req.tolerance':req.tolerance ⋄ ⎕CT}⍬
  ⍝ tolr bounds how small the relative change between accepted iterates
  ⍝ can get before the solver treats it as a stall, not a precision
  ⍝ threshold; loosening it with cfg.tolc only lowers the bar for a
  ⍝ false stall, it never trades precision for speed.
  cfg.tolr←⎕CT
  bounded←2=⎕NC'req.bounds'
  :If bounded
      cfg.lower←1⊃req.bounds
      cfg.upper←2⊃req.bounds
  :EndIf
  :If 2=⎕NC'req.pscale'
      cfg.pscale←req.pscale
  :EndIf
  ⍝ Computation time only, excluding request parsing/dispatch above and
  ⍝ response assembly below - the number RunResult.timing is meant to
  ⍝ compare across backends, not this harness's own request-handling
  ⍝ overhead (#138).
  t0←⎕AI[2]
  r←lower Min(req.x0)cfg
  solveMs←(⎕AI[2])-t0
  gnorm←0.5*⍨+/r.grad×r.grad
  :If Result.MaxIterations r
      status←'MAX_ITER'
  :ElseIf Result.DampingSaturated r
      status←'FAILED'
  :ElseIf bounded
      ⍝ gnorm is the unconstrained gradient norm, genuinely nonzero at
      ⍝ an active-bound optimum; distinguishing a real stall from a
      ⍝ genuine bounded convergence needs a KKT-aware (projected)
      ⍝ gradient check this harness does not yet compute.
      status←'CONVERGED'
  :ElseIf (Result.StalledByEscalation r)∨(Result.StalledByPrecision r)
      status←'STALLED'
  :Else
      status←'CONVERGED'
  :EndIf
  res←⎕NS''
  res.problem_id←req.problem_id
  res.status←status
  res.message←NULL
  res.x_final←r.p
  res.cost_final←r.cost
  res.n_iter←r.iter
  res.n_feval←calls
  res.n_jeval←NULL
  res.n_heval←hcalls
  res.grad_norm_final←gnorm
  res.solve_ms←solveMs
∇
