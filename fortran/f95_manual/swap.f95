program swap_xy
  implicit none
 
  ! local variables
  real :: x, y

  ! external procedures
  interface
    subroutine swap (x, y)
      real, intent(inout) :: x, y
    end subroutine swap
  end interface

  x = 1.5
  y = 3.4
  print*, "x = ", x, "y = ", y
  call swap (x,y)
  print*, "x = ", x, "y = ", y

end program swap_xy

subroutine swap (x, y)
  implicit none

  ! dummy arguments
  real, intent (inout) :: x, y
  
  ! local variables
  real :: buffer

  buffer = x
  x = y
  y = buffer

end subroutine swap
