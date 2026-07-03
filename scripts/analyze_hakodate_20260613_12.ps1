Add-Type -AssemblyName Microsoft.VisualBasic

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutDir = Join-Path $Root 'outputs\analysis'
$HtmlPath = Join-Path $OutDir 'hakodate_20260613_12_shutuba.html'
$CsvPath = Join-Path $Root 'date\raw\全競走馬成績.csv'

function To-Num($s) {
    $v = 0.0
    $clean = ([string]$s) -replace '[^0-9\.\-]', ''
    if ([double]::TryParse($clean, [ref]$v)) { return $v }
    return $null
}

function Find-Index($headers, $name, $nth = 1) {
    $count = 0
    for ($i = 0; $i -lt $headers.Length; $i++) {
        if ($headers[$i] -eq $name) {
            $count++
            if ($count -eq $nth) { return $i }
        }
    }
    return -1
}

function Record-Text($rows) {
    $items = @($rows)
    if ($items.Count -eq 0) { return '0-0-0-0' }
    $w = @($items | Where-Object { $_.着順 -eq 1 }).Count
    $s = @($items | Where-Object { $_.着順 -eq 2 }).Count
    $t = @($items | Where-Object { $_.着順 -eq 3 }).Count
    return "$w-$s-$t-$($items.Count - $w - $s - $t)"
}

$encEuc = [Text.Encoding]::GetEncoding(51932)
$html = $encEuc.GetString([IO.File]::ReadAllBytes($HtmlPath))

$entries = @()
foreach ($m in [regex]::Matches($html, '<tr class="HorseList"[\s\S]*?</tr>')) {
    $row = $m.Value
    $uma = [regex]::Match($row, '<td class="Umaban\d+ Txt_C">(?<n>\d+)</td>').Groups['n'].Value
    $name = [regex]::Match($row, '<span class="HorseName"><a[^>]*title="(?<name>[^"]+)"').Groups['name'].Value
    if (-not $uma -or -not $name) { continue }
    $sexage = ([regex]::Match($row, '<td class="Barei Txt_C">(?<x>[^<]+)</td>').Groups['x'].Value).Trim()
    $weight = ([regex]::Match($row, '<td class="Txt_C">(?<x>\d+\.\d)</td>').Groups['x'].Value).Trim()
    $jockey = ([regex]::Match($row, '<td class="Jockey">[\s\S]*?<a[^>]*>(?<x>.*?)</a>').Groups['x'].Value -replace '<[^>]+>', '').Trim()
    $entries += [pscustomobject]@{ 馬番 = [int]$uma; 馬名 = $name; 性齢 = $sexage; 斤量 = [double]$weight; 騎手 = $jockey }
}

$parser = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser($CsvPath, [Text.Encoding]::GetEncoding(932))
$parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
$parser.SetDelimiters(',')
$parser.HasFieldsEnclosedInQuotes = $true
$headers = $parser.ReadFields()

$ix = @{
    date     = Find-Index $headers '日付'
    place    = Find-Index $headers '場所'
    race     = Find-Index $headers 'Ｒ'
    raceName = Find-Index $headers 'レース名'
    class    = Find-Index $headers 'クラス名'
    horse    = Find-Index $headers '馬名'
    pop      = Find-Index $headers '人気'
    odds     = Find-Index $headers '単勝オッズ'
    finish   = Find-Index $headers '確定着順'
    finish2  = Find-Index $headers '着順'
    surface  = Find-Index $headers '芝・ダ'
    dist     = Find-Index $headers '距離'
    going    = Find-Index $headers '馬場状態'
    time     = Find-Index $headers '走破タイム'
    diff     = Find-Index $headers '着差'
    c1       = Find-Index $headers '1角'
    c2       = Find-Index $headers '2角'
    c3       = Find-Index $headers '3角'
    c4       = Find-Index $headers '4角'
    style    = Find-Index $headers '脚質'
    ave3     = Find-Index $headers 'Ave-3F'
    agari    = Find-Index $headers '上り3F'
    pci      = Find-Index $headers 'PCI'
    pci3     = Find-Index $headers 'PCI3'
    rpci     = Find-Index $headers 'RPCI'
}

$hist = @{}
foreach ($e in $entries) { $hist[$e.馬名] = New-Object System.Collections.Generic.List[object] }

while (-not $parser.EndOfData) {
    $f = $parser.ReadFields()
    if ($f.Length -le $ix.horse) { continue }
    $hn = $f[$ix.horse]
    if (-not $hist.ContainsKey($hn)) { continue }
    $finish = To-Num $f[$ix.finish]
    if ($null -eq $finish) { $finish = To-Num $f[$ix.finish2] }
    $hist[$hn].Add([pscustomobject]@{
        日付S = $f[$ix.date]; 場所 = $f[$ix.place]; R = $f[$ix.race]; レース名 = $f[$ix.raceName]; クラス = $f[$ix.class]
        人気 = (To-Num $f[$ix.pop]); オッズ = (To-Num $f[$ix.odds]); 着順 = $finish; 芝ダ = $f[$ix.surface]; 距離 = (To-Num $f[$ix.dist])
        馬場 = $f[$ix.going].Trim(); 走破 = $f[$ix.time]; 着差 = (To-Num $f[$ix.diff]); C1 = (To-Num $f[$ix.c1]); C2 = (To-Num $f[$ix.c2])
        C3 = (To-Num $f[$ix.c3]); C4 = (To-Num $f[$ix.c4]); 脚質 = $f[$ix.style]; Ave3 = (To-Num $f[$ix.ave3])
        上り = (To-Num $f[$ix.agari]); PCI = (To-Num $f[$ix.pci]); PCI3 = (To-Num $f[$ix.pci3]); RPCI = (To-Num $f[$ix.rpci])
    }) | Out-Null
}
$parser.Close()

$out = @()
foreach ($e in $entries) {
    $h = @($hist[$e.馬名] | Sort-Object 日付S -Descending)
    $d = @($h | Where-Object { $_.芝ダ -eq 'ダ' })
    $d1700 = @($d | Where-Object { $_.距離 -eq 1700 })
    $hakodateD = @($d | Where-Object { $_.場所 -eq '函館' })
    $hakodate1700 = @($d | Where-Object { $_.場所 -eq '函館' -and $_.距離 -eq 1700 })
    $recent = @($h | Select-Object -First 5)
    $score = 0; $plus = @(); $minus = @()

    if ($d1700.Count -ge 2) { $score += 2; $plus += 'ダ1700経験十分' }
    elseif ($d1700.Count -eq 1) { $score += 1; $plus += 'ダ1700経験' }
    else { $minus += 'ダ1700未知' }
    if ($hakodate1700.Count -gt 0) { $score += 2; $plus += '函館ダ1700経験' }
    elseif ($hakodateD.Count -gt 0) { $score += 1; $plus += '函館ダ経験' }
    if (@($d1700 | Where-Object { $_.着順 -ge 1 -and $_.着順 -le 3 }).Count -gt 0) { $score += 2; $plus += 'ダ1700好走' }
    if (@($hakodate1700 | Where-Object { $_.着順 -ge 1 -and $_.着順 -le 3 }).Count -gt 0) { $score += 2; $plus += '函館ダ1700好走' }
    if (@($recent | Select-Object -First 3 | Where-Object { $_.着順 -ge 1 -and $_.着順 -le 3 }).Count -gt 0) { $score += 1; $plus += '近3走内3着以内' }
    if ($e.斤量 -le 55) { $score += 1; $plus += '斤量有利' }

    $avg4 = @($d1700 | Where-Object { $null -ne $_.C4 } | ForEach-Object { $_.C4 } | Measure-Object -Average).Average
    if ($null -eq $avg4) { $avg4 = @($d | Where-Object { $null -ne $_.C4 } | ForEach-Object { $_.C4 } | Measure-Object -Average).Average }
    $styleText = if ($null -eq $avg4) { '不明' } elseif ($avg4 -le 4) { '先行' } elseif ($avg4 -le 8) { '中団' } else { '差し追込' }
    $recentText = ($recent | Select-Object -First 4 | ForEach-Object { "$($_.日付S) $($_.場所)$($_.R)R $($_.レース名) $($_.着順)着/$($_.人気)人気 ダ$($_.距離) 4角$($_.C4) 上り$($_.上り)" }) -join ' / '

    $out += [pscustomobject]@{
        馬番 = $e.馬番; 馬名 = $e.馬名; 性齢 = $e.性齢; 斤量 = $e.斤量; 騎手 = $e.騎手; 適性点 = $score; 脚質 = $styleText
        ダ成績 = (Record-Text $d); ダ1700成績 = (Record-Text $d1700); 函館ダ1700成績 = (Record-Text $hakodate1700)
        平均4角 = if ($null -ne $avg4) { [math]::Round($avg4, 1) } else { $null }
        評価 = if ($plus.Count) { $plus -join '、' } else { '-' }
        懸念 = if ($minus.Count) { $minus -join '、' } else { '-' }
        近走 = $recentText
    }
}

$sorted = $out | Sort-Object @{Expression='適性点';Descending=$true}, @{Expression='馬番';Descending=$false}
$sorted | Format-Table -AutoSize
$sorted | Export-Csv -Path (Join-Path $OutDir 'hakodate_20260613_12_field_analysis.csv') -Encoding UTF8 -NoTypeInformation
$entries | Export-Csv -Path (Join-Path $OutDir 'hakodate_20260613_12_shutuba_clean.csv') -Encoding UTF8 -NoTypeInformation
