#
sudo apt-get install zip

# Java
sudo add-apt-repository -y ppa:webupd8team/java
sudo apt-get update
sudo apt-get install -y oracle-java8-installer
java -version

# maven
wget http://apache.mirror.anlx.net/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.zip
sudo unzip apache-maven-3.3.9-bin.zip
sudo mv apache-maven-3.3.9/ /usr/local/maven/
vim ~/.bashrc
############################################
# maven
export MAVEN_HOME=/usr/local/maven
export PATH=$PATH:$MAVEN_HOME/bin
############################################
source ~/.bashrc
mvn -v

# tomcat
wget http://www.mirrorservice.org/sites/ftp.apache.org/tomcat/tomcat-8/v8.0.32/bin/apache-tomcat-8.0.32.zip
sudo unzip apache-tomcat-8.0.32.zip
sudo mv apache-tomcat-8.0.32/ /usr/local/tomcat/
cd /usr/local/tomcat/
sudo chmod +x bin/*.sh
sudo bin/catalina.sh run
cd /tmp
curl http://localhost:8080

