# Open the built DOCX in Word, update the table of contents (page numbers) and all fields, save it,
# and export a PDF for visual checking. Usage: powershell -File finalize_docx.ps1 <docx> <pdf>
param([string]$Docx, [string]$Pdf)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx)
    foreach ($table in $doc.Tables) {
        $table.Rows.AllowBreakAcrossPages = 0          # no table row split over a page break
        $table.Rows.Item(1).HeadingFormat = -1         # repeat the header row on continuation pages
    }
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    [void]$doc.Fields.Update()
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }   # second pass: page numbers settle after the first
    $doc.Save()
    $doc.ExportAsFixedFormat($Pdf, 17)                           # 17 = wdExportFormatPDF
    "pages: " + $doc.ComputeStatistics(2)                        # 2 = wdStatisticPages
    $doc.Close()
} finally {
    $word.Quit()
}
