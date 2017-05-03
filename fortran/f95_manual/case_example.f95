program case_example
  implicit none
  integer :: n

  ! Ask for a number until 1, 2, or 3 has been enterred
  endless: do
    print*, "Enter a number between 1 and 3: "
    read*, n

    select case (n)
    case (1)
      print*, "You entered 1"
      exit endless
    case (2)
      print*, "You entered 2"
      exit endless
    case (3)
      print*, "You entered 3"
      exit endless
    case default
      print*, "Number is not between 1 and 3"
    end select

  end do endless
end program case_example
