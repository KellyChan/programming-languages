program readout
  implicit none

  integer, parameter :: linelength = 120
  character (len=linelength) :: line
  integer :: ierror

  open (unit=10, file="test.dat", action="read", status="old", iostat=ierror)
  if (ierror /= 0) then
    print*, "Failed to open test.dat"
    stop
  end if

  readfile : do
    read (unit=10, fmt="(a)", iostat=ierror) line
   
    if (ierror < 0) then
      print*, "end of file reached"
      exit readfile

    else if (ierror > 0) then
      print*, "Error during read"
      exit readfile

    else
      ! line has been read successfully
      ! check if it contains energy
      if (index (line, "energy") > 0) then
        print*, "energy found!"
        exit readfile
      end if
    end if
  end do readfile

  close (unit=12)
end program readout
