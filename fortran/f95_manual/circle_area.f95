function circle_area (r)
! this function computes the area of a circle with radius r
  implicit none
  
  ! function result
  real :: circle_area

  ! dummy arguments
  real :: r

  ! local variables
  real :: pi

  pi = 4 * atan (1.0)
  circle_area = pi * r**2

end function circle_area

program main
  a = circle_area (2.0)
  print*, "a = ", a
end program main
