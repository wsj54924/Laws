# Redis 启动脚本
# 用于在 WSL 中启动 Redis 服务器

Write-Host "🚀 启动 Redis 服务器..." -ForegroundColor Green

# 检查 WSL 是否可用
try {
    $wslVersion = wsl --version
    Write-Host "✅ WSL 已安装: $wslVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ WSL 未安装，请先安装 WSL" -ForegroundColor Red
    exit 1
}

# 检查 Redis 是否已安装
Write-Host "`n📦 检查 Redis 安装状态..." -ForegroundColor Yellow
$redisCheck = wsl -e redis-server --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Redis 已安装: $redisCheck" -ForegroundColor Green
} else {
    Write-Host "❌ Redis 未安装，正在安装..." -ForegroundColor Yellow
    wsl -e sudo apt-get update -y
    wsl -e sudo apt-get install redis-server -y
    Write-Host "✅ Redis 安装完成" -ForegroundColor Green
}

# 停止可能正在运行的 Redis 进程
Write-Host "`n🛑 停止现有 Redis 进程..." -ForegroundColor Yellow
wsl -e sudo pkill redis-server 2>$null
Start-Sleep -Seconds 2

# 创建必要的目录
Write-Host "`n📁 创建 Redis 数据目录..." -ForegroundColor Yellow
wsl -e "sudo mkdir -p /var/log/redis /var/lib/redis"

# 设置权限
Write-Host "`n🔐 设置目录权限..." -ForegroundColor Yellow
wsl -e "sudo chmod 755 /var/log/redis /var/lib/redis"

# 复制配置文件
Write-Host "`n📄 复制 Redis 配置文件..." -ForegroundColor Yellow
$configPath = wsl -e wslpath -w "$(pwd)/redis.conf"
wsl -e "sudo cp '$configPath' /etc/redis/redis.conf"

# 启动 Redis 服务器
Write-Host "`n🚀 启动 Redis 服务器..." -ForegroundColor Green
wsl -e sudo redis-server /etc/redis/redis.conf --daemonize yes

# 等待 Redis 启动
Start-Sleep -Seconds 3

# 检查 Redis 是否正在运行
Write-Host "`n🔍 检查 Redis 运行状态..." -ForegroundColor Yellow
$redisStatus = wsl -e redis-cli ping 2>&1
if ($redisStatus -match "PONG") {
    Write-Host "✅ Redis 服务器启动成功！" -ForegroundColor Green
    Write-Host "   Redis 地址: localhost:6379" -ForegroundColor Cyan
    Write-Host "   日志文件: /var/log/redis/redis.log" -ForegroundColor Cyan
    Write-Host "`n💡 提示: 使用 'wsl -e redis-cli' 连接到 Redis" -ForegroundColor Yellow
} else {
    Write-Host "❌ Redis 服务器启动失败" -ForegroundColor Red
    Write-Host "   请检查日志: wsl -e tail -f /var/log/redis/redis.log" -ForegroundColor Yellow
    exit 1
}

# 显示 Redis 信息
Write-Host "`n📊 Redis 服务器信息:" -ForegroundColor Cyan
wsl -e redis-cli info server | Select-String -Pattern "redis_version|os|tcp_port|connected_clients|used_memory_human"

Write-Host "`n✅ Redis 服务器已就绪！" -ForegroundColor Green
Write-Host "`n💡 下一步: 运行 'python test_redis_cache.py' 测试缓存功能" -ForegroundColor Yellow
