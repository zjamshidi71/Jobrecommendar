import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io

# Page configuration
st.set_page_config(
    page_title="Job Recommender System",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .job-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #1f77b4;
    }
    .match-score {
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sample job database (you can replace this with actual data)
@st.cache_data
def load_job_data():
    jobs = pd.DataFrame({
        'job_id': range(1, 21),
        'title': [
            'Senior Data Scientist', 'Machine Learning Engineer', 'Software Developer',
            'Full Stack Developer', 'Data Analyst', 'Business Intelligence Analyst',
            'DevOps Engineer', 'Cloud Solutions Architect', 'Frontend Developer',
            'Backend Developer', 'AI Research Scientist', 'Product Manager',
            'UX Designer', 'Digital Marketing Specialist', 'Cybersecurity Analyst',
            'Systems Administrator', 'Database Administrator', 'Mobile App Developer',
            'Quality Assurance Engineer', 'Technical Writer'
        ],
        'company': [
            'TechCorp', 'AI Innovations', 'CodeWorks', 'WebSolutions', 'DataHub',
            'Analytics Pro', 'CloudTech', 'ScaleSystems', 'DesignLab', 'ServerMasters',
            'ResearchAI', 'ProductCo', 'CreativeStudio', 'MarketingPlus', 'SecureNet',
            'ITManage', 'DatabasePro', 'MobileFirst', 'QualityFirst', 'DocuTech'
        ],
        'location': [
            'San Francisco', 'New York', 'Austin', 'Seattle', 'Boston',
            'Chicago', 'Denver', 'San Francisco', 'Los Angeles', 'Austin',
            'Boston', 'New York', 'San Francisco', 'Chicago', 'Washington DC',
            'Seattle', 'Austin', 'San Francisco', 'Denver', 'Remote'
        ],
        'job_type': [
            'Full-time', 'Full-time', 'Full-time', 'Contract', 'Full-time',
            'Full-time', 'Full-time', 'Full-time', 'Contract', 'Full-time',
            'Full-time', 'Full-time', 'Full-time', 'Part-time', 'Full-time',
            'Full-time', 'Full-time', 'Contract', 'Full-time', 'Remote'
        ],
        'description': [
            'Looking for experienced data scientist with Python, machine learning, and statistical analysis skills. Work on predictive models and data pipelines.',
            'Build and deploy ML models at scale. Required: Python, TensorFlow, PyTorch, AWS. Experience with NLP and computer vision preferred.',
            'Develop robust software solutions using Java, Python, or C++. Strong problem-solving and algorithm design skills required.',
            'Create responsive web applications using React, Node.js, and MongoDB. Experience with modern JavaScript frameworks essential.',
            'Analyze business data, create dashboards, and provide actionable insights. Proficiency in SQL, Excel, and Tableau required.',
            'Design BI solutions and reports using Power BI or Tableau. Strong SQL and data warehousing knowledge needed.',
            'Manage CI/CD pipelines, Docker, Kubernetes. Experience with AWS, Azure, or GCP. Automate infrastructure and deployments.',
            'Design scalable cloud architectures on AWS or Azure. Certifications preferred. Experience with microservices and serverless.',
            'Build beautiful user interfaces with React, Vue, or Angular. Strong CSS, HTML5, and responsive design skills required.',
            'Develop server-side applications using Node.js, Python, or Java. Experience with RESTful APIs and databases essential.',
            'Conduct cutting-edge AI research. PhD preferred. Publications in top-tier conferences. Deep learning and reinforcement learning expertise.',
            'Define product strategy and roadmap. Work with engineering and design teams. Strong communication and analytical skills required.',
            'Create intuitive user experiences. Proficiency in Figma, Sketch, or Adobe XD. Conduct user research and usability testing.',
            'Plan and execute digital marketing campaigns. SEO, SEM, social media marketing experience. Analytics and content creation skills.',
            'Protect systems from cyber threats. Experience with penetration testing, security audits, and incident response required.',
            'Maintain IT infrastructure and servers. Linux/Windows administration. Network configuration and troubleshooting skills needed.',
            'Manage and optimize databases. Experience with MySQL, PostgreSQL, or MongoDB. Performance tuning and backup strategies.',
            'Develop iOS and Android applications. Experience with Swift, Kotlin, or React Native. Strong UI/UX sensibility required.',
            'Ensure software quality through testing. Automation testing experience with Selenium or similar tools. Detail-oriented mindset.',
            'Write technical documentation and user guides. Strong writing skills and ability to explain complex concepts clearly required.'
        ],
        'required_skills': [
            'Python, Machine Learning, Statistics, SQL, Data Visualization',
            'Python, TensorFlow, PyTorch, Machine Learning, AWS, NLP',
            'Java, Python, C++, Algorithms, Data Structures, Problem Solving',
            'React, Node.js, JavaScript, MongoDB, HTML, CSS',
            'SQL, Excel, Tableau, Data Analysis, Statistics',
            'Power BI, Tableau, SQL, Data Warehousing, ETL',
            'Docker, Kubernetes, AWS, CI/CD, Linux, Automation',
            'AWS, Azure, Cloud Architecture, Microservices, Serverless',
            'React, Vue, Angular, HTML5, CSS3, JavaScript, Responsive Design',
            'Node.js, Python, Java, REST API, SQL, NoSQL',
            'Deep Learning, PyTorch, TensorFlow, Research, NLP, Computer Vision',
            'Product Strategy, Agile, Communication, Analytics, Leadership',
            'Figma, Sketch, Adobe XD, User Research, Prototyping, Wireframing',
            'SEO, SEM, Google Analytics, Social Media, Content Marketing',
            'Security, Penetration Testing, Network Security, Incident Response',
            'Linux, Windows Server, Networking, Troubleshooting, Active Directory',
            'MySQL, PostgreSQL, MongoDB, Database Design, Performance Tuning',
            'Swift, Kotlin, React Native, iOS, Android, Mobile UI/UX',
            'Selenium, Testing, QA Automation, Python, Java, Bug Tracking',
            'Technical Writing, Documentation, Communication, API Documentation'
        ],
        'salary_range': [
            '$120k - $180k', '$130k - $190k', '$90k - $140k', '$80k - $120k',
            '$70k - $110k', '$75k - $115k', '$110k - $160k', '$140k - $200k',
            '$85k - $130k', '$95k - $145k', '$150k - $220k', '$120k - $170k',
            '$80k - $120k', '$60k - $90k', '$100k - $150k', '$70k - $110k',
            '$85k - $135k', '$95k - $145k', '$80k - $125k', '$65k - $100k'
        ]
    })
    return jobs

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# Function to calculate job match scores
def calculate_match_scores(resume_text, jobs_df):
    # Combine job title, description, and required skills for better matching
    jobs_df['combined_text'] = (
        jobs_df['title'] + ' ' + 
        jobs_df['description'] + ' ' + 
        jobs_df['required_skills']
    )
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform on job descriptions
    all_texts = [resume_text] + jobs_df['combined_text'].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity between resume and all jobs
    resume_vector = tfidf_matrix[0:1]
    job_vectors = tfidf_matrix[1:]
    
    similarity_scores = cosine_similarity(resume_vector, job_vectors)[0]
    
    # Convert to percentage and add to dataframe
    jobs_df['match_score'] = (similarity_scores * 100).round(2)
    
    return jobs_df.sort_values('match_score', ascending=False)

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">💼 Job Recommender System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filter Options")
    
    # Load job data
    jobs_df = load_job_data()
    
    # Sidebar filters
    location_filter = st.sidebar.multiselect(
        "📍 Location",
        options=["All"] + sorted(jobs_df['location'].unique().tolist()),
        default=["All"]
    )
    
    job_type_filter = st.sidebar.multiselect(
        "💼 Job Type",
        options=["All"] + sorted(jobs_df['job_type'].unique().tolist()),
        default=["All"]
    )
    
    min_match_score = st.sidebar.slider(
        "📊 Minimum Match Score (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Tip:** Upload your resume to get personalized job recommendations "
        "based on your skills and experience!"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Upload Your Resume")
        uploaded_file = st.file_uploader(
            "Choose a PDF or text file",
            type=['pdf', 'txt'],
            help="Upload your resume in PDF or TXT format"
        )
    
    with col2:
        st.subheader("✏️ Or Paste Your Resume")
        resume_text_input = st.text_area(
            "Paste your resume text here",
            height=200,
            help="Copy and paste your resume content"
        )
    
    # Process resume
    resume_text = ""
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")
        
        if resume_text:
            st.success("✅ Resume uploaded successfully!")
    elif resume_text_input:
        resume_text = resume_text_input
        st.success("✅ Resume text received!")
    
    # Display results
    st.markdown("---")
    
    if resume_text:
        with st.spinner("🔄 Analyzing your resume and finding best matches..."):
            # Calculate match scores
            matched_jobs = calculate_match_scores(resume_text, jobs_df.copy())
            
            # Apply filters
            if "All" not in location_filter:
                matched_jobs = matched_jobs[matched_jobs['location'].isin(location_filter)]
            
            if "All" not in job_type_filter:
                matched_jobs = matched_jobs[matched_jobs['job_type'].isin(job_type_filter)]
            
            matched_jobs = matched_jobs[matched_jobs['match_score'] >= min_match_score]
            
            # Display results
            st.subheader(f"🎯 Top Job Matches ({len(matched_jobs)} found)")
            
            if len(matched_jobs) == 0:
                st.warning("No jobs found matching your criteria. Try adjusting the filters.")
            else:
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Match", f"{matched_jobs['match_score'].mean():.1f}%")
                with col2:
                    st.metric("Best Match", f"{matched_jobs['match_score'].max():.1f}%")
                with col3:
                    st.metric("Total Jobs", len(matched_jobs))
                
                st.markdown("---")
                
                # Display job cards
                for idx, job in matched_jobs.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"### {job['title']}")
                            st.markdown(f"**{job['company']}** | {job['location']} | {job['job_type']}")
                        
                        with col2:
                            score_color = "green" if job['match_score'] >= 70 else "orange" if job['match_score'] >= 50 else "red"
                            st.markdown(
                                f"<div style='text-align: center; padding: 10px; background-color: {score_color}; "
                                f"color: white; border-radius: 10px; font-weight: bold;'>"
                                f"{job['match_score']:.1f}% Match</div>",
                                unsafe_allow_html=True
                            )
                        
                        st.markdown(f"**Description:** {job['description']}")
                        st.markdown(f"**Required Skills:** {job['required_skills']}")
                        st.markdown(f"**💰 Salary Range:** {job['salary_range']}")
                        
                        if st.button(f"Apply Now", key=f"apply_{job['job_id']}"):
                            st.success(f"Application initiated for {job['title']} at {job['company']}!")
                        
                        st.markdown("---")
    
    else:
        # Show all jobs when no resume is uploaded
        st.subheader("📋 Available Jobs")
        st.info("👆 Upload your resume or paste your resume text to get personalized recommendations!")
        
        # Apply filters
        display_jobs = jobs_df.copy()
        
        if "All" not in location_filter:
            display_jobs = display_jobs[display_jobs['location'].isin(location_filter)]
        
        if "All" not in job_type_filter:
            display_jobs = display_jobs[display_jobs['job_type'].isin(job_type_filter)]
        
        st.write(f"**Showing {len(display_jobs)} jobs**")
        
        for idx, job in display_jobs.iterrows():
            with st.expander(f"{job['title']} at {job['company']} - {job['location']}"):
                st.markdown(f"**Job Type:** {job['job_type']}")
                st.markdown(f"**Description:** {job['description']}")
                st.markdown(f"**Required Skills:** {job['required_skills']}")
                st.markdown(f"**💰 Salary Range:** {job['salary_range']}")
                
                if st.button(f"Learn More", key=f"learn_{job['job_id']}"):
                    st.info(f"More details about {job['title']} would be shown here.")

if __name__ == "__main__":
    main()
