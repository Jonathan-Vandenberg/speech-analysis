#!/bin/bash
set -e

echo "🚀 Setting up Speech Analyser API Server on DigitalOcean"

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
echo "🐙 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Nginx
echo "🌐 Installing Nginx..."
sudo apt install -y nginx

# Install Certbot for SSL
echo "🔐 Installing Certbot..."
sudo apt install -y certbot python3-certbot-nginx

# Install doctl (DigitalOcean CLI)
echo "🔧 Installing doctl..."
cd ~
wget https://github.com/digitalocean/doctl/releases/download/v1.104.0/doctl-1.104.0-linux-amd64.tar.gz
tar xf ~/doctl-1.104.0-linux-amd64.tar.gz
sudo mv ~/doctl /usr/local/bin
rm ~/doctl-1.104.0-linux-amd64.tar.gz

# Create app directory
echo "📁 Creating application directory..."
sudo mkdir -p /var/www/speech-analyser
sudo chown $USER:$USER /var/www/speech-analyser

# Setup Nginx configuration
echo "⚙️ Setting up Nginx configuration..."
sudo cp /var/www/speech-analyser/nginx.conf /etc/nginx/sites-available/api.speechanalyser.com
sudo ln -sf /etc/nginx/sites-available/api.speechanalyser.com /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
sudo nginx -t

# Start and enable services
echo "🔄 Starting services..."
sudo systemctl start nginx
sudo systemctl enable nginx
sudo systemctl start docker
sudo systemctl enable docker

# Setup SSL certificate
echo "🔐 Setting up SSL certificate..."
echo "Please make sure your domain is pointing to this server before continuing."
read -p "Press Enter when ready to continue with SSL setup..."

sudo certbot --nginx -d api.speechanalyser.com --non-interactive --agree-tos --email your-email@example.com

# Setup firewall
echo "🔥 Configuring firewall..."
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

# Create deployment directory
mkdir -p ~/deployments
cd ~/deployments

# Setup doctl authentication
echo "🔑 Setting up DigitalOcean CLI authentication..."
echo "Please run: doctl auth init"
echo "Use your DigitalOcean API token"

# Create systemd service for the API (optional - as backup to Docker)
sudo tee /etc/systemd/system/speech-analyser-api.service > /dev/null <<EOF
[Unit]
Description=Speech Analyser API
After=network.target

[Service]
Type=exec
User=$USER
WorkingDirectory=/var/www/speech-analyser
ExecStart=/usr/local/bin/docker run --rm -p 8000:8000 --name speech-analyser-api registry.digitalocean.com/speech-analyser/api:latest
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Server setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Configure your GitHub secrets:"
echo "   - DIGITALOCEAN_ACCESS_TOKEN: Your DO API token"
echo "   - DROPLET_HOST: This server's IP address"
echo "   - DROPLET_USER: $USER"
echo "   - DROPLET_SSH_KEY: Your private SSH key"
echo "   - SUPABASE_URL: Your Supabase project URL"
echo "   - SUPABASE_ANON_KEY: Your Supabase service_role key"
echo "   - OPENAI_API_KEY: Your OpenAI API key"
echo ""
echo "2. Update the email in the SSL setup command above"
echo "3. Point your domain api.speechanalyser.com to this server"
echo "4. Push to main branch to trigger deployment"
echo ""
echo "🌐 Your API will be available at: https://api.speechanalyser.com"
