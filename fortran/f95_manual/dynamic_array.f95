program dynamic_array
  implicit none

  real, dimension (:,:), allocatable :: a
  integer :: dim1, dim2
  integer :: i, j

  print*, "Give dimensions dim1 and dim2: "
  read*, dim1, dim2

  ! now that the size of a is know, allocate memory for it
  allocate (a(dim1, dim2))

  do i = 1, dim2
    do j = 1, dim1
      a(j, i) = i * j
      print*, "a(", j, ",", i, ") = ", a(j, i)
    end do
  end do

  deallocate(a)

end program dynamic_array
