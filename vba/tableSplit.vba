Project: Market Research (table splitted)

Programmer: Kelly Chan
Date Created: Aug 30 2013

-----------------------------------------------------------------------------------

Sub split()

Dim lastrow As Long, LastCol As Integer, i As Long, iStart As Long, iEnd As Long, sheetName As Long
Dim ws As Worksheet

With ActiveSheet

    lastrow = .Cells(Rows.Count, "A").End(xlUp).Row
    LastCol = .Cells(12, Columns.Count).End(xlToLeft).Column
    .Range(.Cells(1, 1), Cells(lastrow, LastCol)).Select
    
    iStart = 2
    
    For i = 2 To lastrow
        If .Range("A" & i).Value = "#page" Then
        
            iEnd = i
            sheetName = iStart + 2
            
            Sheets.Add after:=Sheets(Sheets.Count)
            Set ws = ActiveSheet
            On Error Resume Next
            ws.Name = .Range("A" & sheetName).Value
            On Error GoTo 0
            
            ws.Range(Cells(1, 1), Cells(1, LastCol)).Value = .Range(.Cells(iStart, 1), .Cells(iStart, LastCol)).Value
            .Range(.Cells(iStart, 1), .Cells(iEnd, LastCol)).Copy Destination:=ws.Range("A1")

            iStart = iEnd + 1

        End If
    Next i
End With

End Sub
