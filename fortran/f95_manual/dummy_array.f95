program dummy_array

  implicit none
 
  integer, dimension (10) :: a
  
  call fill_array(a)
  print*, a

end program dummy_array


subroutine fill_array (a)
  implicit none

  ! dummy arguments
  integer, dimension (10), intent (out) :: a

  ! local variables
  integer :: i

  do i = 1, 10
    a(i) = i
  end do

end subroutine fill_array
