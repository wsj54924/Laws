# Redis 停止脚本
# 用于在 WSL 中停止 Redis 服务器

Write-Host "🛑 停止 Redis 服务器..." -ForegroundColor Yellow

# 停止 Redis 进程
Write-Host "`n🛑 停止 Redis 进程..." -ForegroundColor Yellow
wsl -e sudo pkill redis-server 2>$null

# 等待进程停止
Start-Sleep -Seconds 2

# 检查 Redis 是否已停止
Write-Host "`n🔍 检查 Redis 运行状态..." -ForegroundColor Yellow
$redisStatus = wsl -e redis-cli ping 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "✅ Redis 服务器已停止" -ForegroundColor Green
} else {
    Write-Host "❌ Redis 服务器仍在运行" -ForegroundColor Red
    Write-Host "   尝试强制停止..." -ForegroundColor Yellow
    wsl -e sudo killall -9 redis-server 2>$null
    Start-Sleep -Seconds 1
    
    $redisStatus = wsl -e redis-cli ping 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✅ Redis 服务器已强制停止" -ForegroundColor Green
    } else {
        Write-Host "❌ 无法停止 Redis 服务器" -ForegroundColor Red
    }
}

Write-Host "`n✅ Redis 停止脚本执行完成" -ForegroundColor Green
