const analyzeUrl = (url) => {
  // Extract basic features from URL
  const urlLength = url.length
  const hasHttps = url.toLowerCase().startsWith("https://")
  const hasHttp = url.toLowerCase().startsWith("http://") && !hasHttps
  const specialChars = (url.match(/[!@#$%^&*()_\+\-=\[\]{};':\\|,.<>\/?~`]/g) || []).length
  const hasIp = /^\d+\.\d+\.\d+\.\d+/.test(url.split("/")[2])
  const domain = url.replace(/^https?:\/\//, "").split("/")[0]
  const path = url.replace(/^https?:\/\/[^\/]+/, "").split("?")[0]
  const pathLength = path.length
  const subdomainCount = domain.split(".").length - 2
  const tld = domain.split(".").pop()
  const queryString = url.includes("?") ? "yes" : "no"
  const fragment = url.includes("#") ? "yes" : "no"

  // Phishing risk assessment
  let riskScore = 0
  const factors = []

  if (!hasHttps && hasHttp) {
    riskScore += 25
    factors.push("No HTTPS encryption")
  }
  if (urlLength > 75) {
    riskScore += 20
    factors.push("Very long URL")
  }
  if (urlLength > 100) {
    riskScore += 15
  }
  if (specialChars > 10) {
    riskScore += 20
    factors.push("Many special characters")
  }
  if (hasIp) {
    riskScore += 30
    factors.push("Uses IP address instead of domain")
  }
  if (subdomainCount > 2) {
    riskScore += 10
    factors.push("Many subdomains")
  }
  if (queryString === "yes") {
    riskScore += 5
  }
  if (fragment === "yes") {
    riskScore += 5
  }

  const isPhishing = riskScore >= 50
  const confidence = Math.min(95, 50 + riskScore)

  return {
    url,
    isPhishing,
    confidence: Math.round(confidence),
    riskScore,
    factors,
    features: {
      urlLength,
      hasHttps,
      specialChars,
      hasIp,
      subdomainCount,
      tld,
      pathLength,
      queryString,
      fragment,
    },
  }
}

export default function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" })
  }

  try {
    const { url } = req.body
    if (!url) {
      return res.status(400).json({ error: "URL is required" })
    }

    const result = analyzeUrl(url)
    res.status(200).json(result)
  } catch (error) {
    console.error("Analysis error:", error)
    res.status(500).json({ error: "Internal server error" })
  }
}