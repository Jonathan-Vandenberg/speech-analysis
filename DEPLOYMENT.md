# 🚀 DigitalOcean Deployment Guide

Deploy your Speech Analysis API to DigitalOcean with automatic deployments on every push to main.

## 🌐 Domain Setup (Namecheap + DigitalOcean)

### Option 1: Use DigitalOcean DNS (Recommended)

**In Namecheap:**
1. Go to Domain List → Manage → Advanced DNS
2. Change Nameservers to:
   ```
   ns1.digitalocean.com
   ns2.digitalocean.com
   ns3.digitalocean.com
   ```

**In DigitalOcean:**
1. Go to Networking → Domains
2. Add domain: `speechanalyser.com`
3. Create DNS records:
   ```
   A     @     YOUR_DROPLET_IP     3600
   A     api   YOUR_DROPLET_IP     3600
   A     www   YOUR_DROPLET_IP     3600
   ```

### Option 2: Use Namecheap DNS

**In Namecheap Advanced DNS:**
```
Type    Host    Value              TTL
A       @       YOUR_DROPLET_IP    Automatic
A       api     YOUR_DROPLET_IP    Automatic
A       www     YOUR_DROPLET_IP    Automatic
```

## 🖥️ Server Setup

### 1. Create DigitalOcean Droplet

**Recommended specs:**
- **OS**: Ubuntu 22.04 LTS
- **Size**: 2 GB RAM, 1 vCPU ($12/month)
- **Storage**: 50 GB SSD
- **Region**: Choose closest to your users

### 2. Initial Server Configuration

```bash
# SSH into your droplet
ssh root@YOUR_DROPLET_IP

# Create user and add to sudo
adduser deploy
usermod -aG sudo deploy
su - deploy

# Copy your public SSH key
mkdir ~/.ssh
echo "YOUR_SSH_PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### 3. Run Server Setup Script

```bash
# Clone your repository
git clone https://github.com/Jonathan-Vandenberg/speech-analysis.git
cd speech-analysis

# Make setup script executable
chmod +x server-setup.sh

# Run setup (this installs Docker, Nginx, SSL, etc.)
./server-setup.sh
```

### 4. Configure doctl (DigitalOcean CLI)

```bash
# Authenticate with DigitalOcean
doctl auth init
# Enter your DigitalOcean API token

# Create container registry
doctl registry create speech-analyser
```

## 🔐 GitHub Secrets Configuration

Go to your repository → Settings → Secrets and variables → Actions

Add these secrets:

```bash
# DigitalOcean
DIGITALOCEAN_ACCESS_TOKEN=dop_v1_xxxxxxxxxxxxx
DROPLET_HOST=YOUR_DROPLET_IP
DROPLET_USER=deploy
DROPLET_SSH_KEY=-----BEGIN OPENSSH PRIVATE KEY-----
...your private SSH key...
-----END OPENSSH PRIVATE KEY-----

# Database
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# AI Service
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

### How to get your SSH private key:

```bash
# On your local machine
cat ~/.ssh/id_rsa

# If you don't have SSH keys, create them:
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

## 🔄 Deployment Process

### Automatic Deployment

Every push to `main` branch will:

1. 🏗️ **Build** Docker image
2. 📤 **Push** to DigitalOcean Container Registry  
3. 🚀 **Deploy** to your droplet
4. ♻️ **Restart** the API service
5. 🧹 **Clean up** old images

### Manual Deployment

```bash
# On your server
cd ~/deployments
doctl registry login
docker pull registry.digitalocean.com/speech-analyser/api:latest

# Stop and remove old container
docker stop speech-analyser-api || true
docker rm speech-analyser-api || true

# Run new container
docker run -d \
  --name speech-analyser-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e SUPABASE_URL="https://xxxxx.supabase.co" \
  -e SUPABASE_ANON_KEY="your_service_role_key" \
  -e OPENAI_API_KEY="sk-xxxxx" \
  registry.digitalocean.com/speech-analyser/api:latest
```

## 🌐 SSL Certificate Setup

The server setup script includes SSL via Let's Encrypt:

```bash
# This is run automatically in the setup script
sudo certbot --nginx -d api.speechanalyser.com
```

**Update the email in `server-setup.sh` before running!**

## ✅ Verification

After deployment, verify everything works:

### 1. Health Check
```bash
curl https://api.speechanalyser.com/healthz
# Should return: {"status":"ok","database":"connected","version":"1.0.0"}
```

### 2. API Documentation
Visit: https://api.speechanalyser.com/docs

### 3. Test API Endpoint
```bash
curl -X POST "https://api.speechanalyser.com/api/admin/keys" \
  -F "description=Test Key" \
  -F "minute_limit=10"
```

## 🔧 Troubleshooting

### Check Container Logs
```bash
docker logs speech-analyser-api
```

### Check Nginx Status
```bash
sudo systemctl status nginx
sudo nginx -t  # Test configuration
```

### Check SSL Certificate
```bash
sudo certbot certificates
```

### Restart Services
```bash
# Restart API
docker restart speech-analyser-api

# Restart Nginx
sudo systemctl restart nginx
```

### Domain Not Resolving
```bash
# Check DNS propagation
dig api.speechanalyser.com
nslookup api.speechanalyser.com
```

## 📊 Monitoring

### Server Resources
```bash
# CPU and Memory usage
htop

# Disk usage
df -h

# Docker stats
docker stats speech-analyser-api
```

### API Logs
```bash
# Live logs
docker logs -f speech-analyser-api

# Last 100 lines
docker logs --tail 100 speech-analyser-api
```

## 🔄 Updates

### Code Updates
Just push to main branch - automatic deployment will handle it!

### Server Updates
```bash
sudo apt update && sudo apt upgrade -y
sudo reboot  # If kernel updates
```

### SSL Renewal
Certbot auto-renews, but to test:
```bash
sudo certbot renew --dry-run
```

## 🎯 Final URLs

After successful deployment:

- **🔗 API**: https://api.speechanalyser.com
- **📚 Documentation**: https://api.speechanalyser.com/docs
- **📋 ReDoc**: https://api.speechanalyser.com/redoc  
- **🏥 Health**: https://api.speechanalyser.com/healthz
- **📖 GitHub Pages**: https://jonathan-vandenberg.github.io/speech-analysis

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review GitHub Actions logs for deployment issues
3. Check server logs with the commands provided
4. Verify DNS propagation with online tools

---

**🎉 Your Speech Analysis API is now live and ready for production use!**
