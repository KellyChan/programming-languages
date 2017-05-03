module student_class

implicit none

  private
  public :: create_student, get_mark

  type student_data
    character (len=50) :: name
    real :: mark
  end type student_data

  type (student_data), dimension (100) :: student

contains

  subroutine create_student (student_n, name, mark)
    ! here some code to set the anme and mark of a student
  end subroutine create_student

  subroutine get_mark (name, mark)
    ! dummy arguments
    character (len=*), intent (in) :: name
    real, intent (out) :: mark

    ! local variables
    integer :: i

    do i = 1, 100
      if (student(i) % name == name) then
        mark = student(i) % mark
      end if
    end do

  end subroutine get_mark

end module student_class


program student_list
  use student_class
  implicit none

  real :: mark

  call create_student (1, "Peter Peterson", 8.5)
  call create_student (2, "John Johnson", 6.3)

  call get_mark ("Peter Peterson", mark)
  print*, "Peter Peterson: ", mark
end program student_list
