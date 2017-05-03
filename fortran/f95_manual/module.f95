module constants

implicit none

real, parameter :: pi = 3.1415926536
real, parameter :: e = 2.7182818285

contains

    subroutine show_consts()
      print*, "pi = ", pi
      print*, "e = ", e
    end subroutine show_consts  

end module constants


program module_example

  use constants
  implicit none

  real :: twopi

  twopi = 2 * pi
  call show_consts()
  print*, "twopi = ", twopi

end program module_example
